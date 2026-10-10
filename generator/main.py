import multiprocessing
import os
import random
import struct
import time
from collections import Counter

import chess
import chess.engine
import chess.pgn
import tomllib
from chess import Board
from chess.engine import SimpleEngine




def load_map(path) -> list:
    openings = []
    with open(path, 'r', encoding='utf8') as book:
        while (game := chess.pgn.read_game(book)) is not None:
            openings.append(game.end().board().fen()) 

        
    print("Done with openings!")
    return openings



def result_distribution(path: str, chunk_records: int = 100_000) -> Counter:
    counts = Counter()
    with open(path, "rb") as f:
        while chunk := f.read(RECORD.size * chunk_records):
            usable = len(chunk) - len(chunk) % RECORD.size
            for _, _, _, _, result in RECORD.iter_unpack(chunk[:usable]):
                counts[result] += 1

    total = sum(counts.values())
    names = {0: "white wins", 1: "black wins", 2: "draws"}
    for r in sorted(counts):
        print(f"{names.get(r, r)}: {counts[r]} ({100 * counts[r] / total:.1f}%)")
    print(f"total: {total}")
    return counts



def play_random_moves(board: Board, config) -> bool:
    for _ in range(random.randint(config["openings"]["random_plies"][0], config["openings"]["random_plies"][1])):
        if board.is_game_over():
            return False
        board.push(random.choice(list(board.legal_moves)))
    return not board.is_game_over()



RECORD = struct.Struct("<Q16sBhB4x")  # 32 bytes per position
RESULT_OFFSET = 27   # 8 occupancy + 16 pieces + 1 side to move + 2 score

def data_encoder(board: Board, score: int, result: int) -> bytes:
    occ = board.occupied
    nibbles = []
    for sq in chess.scan_forward(occ):
        piece = board.piece_at(sq)
        code = piece.piece_type - 1          # pawn..king -> 0..5

        if piece.color == chess.BLACK:
            code |= 8                        # set the color bit

        nibbles.append(code)
        
    nibbles += [0] * (32 - len(nibbles))     # pad up to 32 slots
    packed = bytes(nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, 32, 2))
    stm = 0 if board.turn == chess.WHITE else 1
    return RECORD.pack(occ, packed, stm, score, result)

def encode_result(record: bytearray, result: int) -> None:
    record[RESULT_OFFSET] = result


def data_decoder(rec: bytes):
    occ, packed, stm, score, result = RECORD.unpack(rec)

    nibbles = []
    for byte in packed:
        nibbles.append(byte & 15)    # low nibble first, matching the encoder
        nibbles.append(byte >> 4)

    board = Board(None)              # empty board
    for i, sq in enumerate(chess.scan_forward(occ)):
        n = nibbles[i]
        color = chess.BLACK if n & 8 else chess.WHITE
        board.set_piece_at(sq, chess.Piece((n & 7) + 1, color))

    board.turn = chess.WHITE if stm == 0 else chess.BLACK
    return board, score, result



def run_worker(worker_id: int, config, openings: list, export_path: str) -> int:
    random.seed()  # fresh seed per process

    workers = config["generation"]["workers"]
    target = -(-config["generation"]["data_count"] // workers)  # rounds up
    step = max(1, target // config["generation"]["checkpoints"])
    next_checkpoint = step

    engine = chess.engine.SimpleEngine.popen_uci(config["engine"]["path"])
    board = chess.Board()
    training_data = []
    total = 0
    start = time.perf_counter()

    with open(export_path, "ab") as export_game_file:
        while total < target:
            game = play_game(board, engine, openings, config)
            training_data.extend(game)
            total += len(game)

            if total >= next_checkpoint:
                export_game_file.write(b"".join(training_data))
                export_game_file.flush()
                training_data.clear()
                print(f"[w{worker_id}] {total}/{target}, {time.perf_counter() - start:.1f}s")
                next_checkpoint += step

        # write whatever is left after the last checkpoint
        export_game_file.write(b"".join(training_data))
        training_data.clear()

    engine.quit()
    print(f"[w{worker_id}] done: {total} in {time.perf_counter() - start:.1f}s")
    return total


def play_game(board : Board, engine : SimpleEngine, openings : list, config) -> list:
    # set up opening
    game_id = object()
    game_data = []
    # 0 = WHITE WIN, 1 = BLACK WIN, 2 = DRAW
    game_result = 2

    # a couple random moves.
    while True:
        board.set_fen(random.choice(openings))
        if play_random_moves(board, config):
            break

    while not board.is_game_over():
        result = engine.play(board, chess.engine.Limit(nodes=config["engine"]["nodes"]), info=chess.engine.INFO_SCORE, game=game_id)

        # check if move is data gen worthy
        # quit early, skip long positions update game state.
        white_score = result.info["score"].pov(chess.WHITE).score()

        if (abs(white_score) > config["adjudication"]["win_cp"]):
            if white_score > config["adjudication"]["win_cp"]:
                game_result = 0
            elif white_score < -config["adjudication"]["win_cp"]:
                game_result = 1
            break

        # dont add moves that are checking, in check, captures or promotions into data gen.
        if not (board.gives_check(result.move) or board.is_capture(result.move) or 
                 result.move.promotion is not None or board.is_check()):
            
            # Store encoded data
            game_data.append(bytearray(data_encoder(board, white_score, game_result)))

        # make move
        board.push(result.move)

    if game_result != 2:
        for item in game_data:
            encode_result(item, game_result)

    return game_data


def build_export_path(config, worker_id: int) -> str:
    e = config["export"]
    nodes = short_count(config["engine"]["nodes"])
    size = short_count(config["generation"]["data_count"])
    return f"{e['path']}{e['name']}_g{e['generation']}_{nodes}_{size}_w{worker_id}{e['extension']}"

    
def short_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    if n >= 1_000:
        return f"{n / 1_000:g}k"
    return str(n)


if __name__ == "__main__":
    with open("./config.toml", "rb") as f:
        config = tomllib.load(f)

    os.makedirs(config["export"]["path"], exist_ok=True)
    openings = load_map(config["openings"]["folder"] + config["openings"]["path"])
    workers = config["generation"]["workers"]
    paths = [build_export_path(config, i) for i in range(workers)]

    with multiprocessing.Pool(workers) as pool:
        totals = pool.starmap(run_worker, [(i, config, openings, paths[i]) for i in range(workers)])

    print(f"Finished, total samples: {sum(totals)}")
    result_distribution(paths)