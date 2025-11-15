import chess
import numpy as np
import chess.pgn
from tqdm import tqdm
import random
import torch
import json
from pathlib import Path

def build_vocabulary():
    move_to_idx = {}
    
    idx = 0
    for from_square in range(64):
        for to_square in range(64):
            move = chess.Move(from_square, to_square)
            move_uci = move.uci()
            
            if move_uci not in move_to_idx:
                move_to_idx[move_uci] = idx
                idx += 1
                
    for from_file in range(8):
        from_square = 48 + from_file 
        
        to_square = 56 + from_file 
        
        for promotion_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            move = chess.Move(from_square, to_square, promotion=promotion_piece)
            move_uci = move.uci() 
            
            if move_uci not in move_to_idx:
                move_to_idx[move_uci] = idx
                idx += 1
        
        if from_file > 0:
            to_square = 56 + (from_file - 1)
            
            for promotion_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(from_square, to_square, promotion=promotion_piece)
                move_uci = move.uci() 
                
                if move_uci not in move_to_idx:
                    move_to_idx[move_uci] = idx
                    idx += 1
        
        if from_file < 7:
            to_square = 56 + (from_file + 1)  
            
            for promotion_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(from_square, to_square, promotion=promotion_piece)
                move_uci = move.uci()  
                
                if move_uci not in move_to_idx:
                    move_to_idx[move_uci] = idx
                    idx += 1
    
    for from_file in range(8):
        from_square = 8 + from_file 
        
        to_square = 0 + from_file 
        
        for promotion_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            move = chess.Move(from_square, to_square, promotion=promotion_piece)
            move_uci = move.uci()  
            
            if move_uci not in move_to_idx:
                move_to_idx[move_uci] = idx
                idx += 1
        
        if from_file > 0:
            to_square = 0 + (from_file - 1)
            
            for promotion_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(from_square, to_square, promotion=promotion_piece)
                move_uci = move.uci()
                
                if move_uci not in move_to_idx:
                    move_to_idx[move_uci] = idx
                    idx += 1
        
        if from_file < 7:
            to_square = 0 + (from_file + 1)
            
            for promotion_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(from_square, to_square, promotion=promotion_piece)
                move_uci = move.uci()
                
                if move_uci not in move_to_idx:
                    move_to_idx[move_uci] = idx
                    idx += 1
    
    print(f"After promotions: {len(move_to_idx)} total moves")
    
    return move_to_idx
                
def board_to_tensor(board):
    tensor = np.zeros((17, 8, 8), dtype=np.float32) 
    
    piece_to_channel = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            rank = square // 8
            file = square % 8
            
            channel = piece_to_channel[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6  # Black pieces in channels 6-11
            
            tensor[channel, rank, file] = 1.0

    # Castling rights (channels 12-15)
    tensor[12, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    tensor[13, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    tensor[14, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    tensor[15, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    
    # En passant square (channel 16)
    if board.ep_square is not None:
        rank = board.ep_square // 8
        file = board.ep_square % 8
        tensor[16, rank, file] = 1.0
    
    return tensor

def process_pgn(pgn_path, move_to_idx, max_positions=250000, opening_sample_rate = 0.2):
    positions = []
    
    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
        game_count = 0
        
        with tqdm(total=max_positions, desc="Extracting positions") as pbar:
            while len(positions) < max_positions:
                game = chess.pgn.read_game(pgn_file)
                
                if game is None:
                    print("\nReached end of PGN file")
                    break
                
                game_id = game_count
                
                # Extract positions from this game
                board = game.board()
                move_num = 0
                
                for move in game.mainline_moves():
                    move_num += 1
                    
                    # Downsample opening moves (first 5 moves)
                    if move_num <= 5:
                        if random.random() > opening_sample_rate:
                            board.push(move)
                            continue
                    
                    # Convert board to tensor
                    board_tensor = board_to_tensor(board)
                    move_uci = move.uci()
                    
                    # Check if move is in vocabulary
                    if move_uci in move_to_idx:
                        move_idx = move_to_idx[move_uci]
                        
                        # Original position
                        positions.append((board_tensor, move_idx, game_id))
                        pbar.update(1)
                        
                        # Horizontal flip augmentation
                        flipped_tensor = np.flip(board_tensor, axis=2).copy()
                        
                        # Flip move coordinates
                        from_square = move.from_square
                        to_square = move.to_square
                        
                        from_file = from_square % 8
                        from_rank = from_square // 8
                        to_file = to_square % 8
                        to_rank = to_square // 8
                        
                        # Flip files (a↔h, b↔g, etc.)
                        flipped_from = from_rank * 8 + (7 - from_file)
                        flipped_to = to_rank * 8 + (7 - to_file)
                        
                        flipped_move = chess.Move(flipped_from, flipped_to)
                        
                        # Handle promotions
                        if move.promotion:
                            flipped_move = chess.Move(flipped_from, flipped_to, 
                                                     promotion=move.promotion)
                        
                        flipped_uci = flipped_move.uci()
                        
                        if flipped_uci in move_to_idx:
                            flipped_idx = move_to_idx[flipped_uci]
                            positions.append((flipped_tensor, flipped_idx, game_id))
                            pbar.update(1)
                        
                        if len(positions) >= max_positions:
                            break
                    
                    board.push(move)
                
                game_count += 1
                
                if game_count % 100 == 0:
                    print(f"\nProcessed {game_count} games, {len(positions)} positions")
    
    print(f"\n✓ Extracted {len(positions)} positions from {game_count} games")
    return positions
    
def save_data(positions, move_to_idx, output_dir='data/processed', val_ratio=0.1, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(positions)} positions...")
    
    # --- Split positions by game_id ---
    random.seed(seed)
    game_ids = list(set([p[2] for p in positions]))  # extract unique game IDs
    random.shuffle(game_ids)
    
    val_count = int(len(game_ids) * val_ratio)
    val_game_ids = set(game_ids[:val_count])
    
    train_positions = [p for p in positions if p[2] not in val_game_ids]
    val_positions = [p for p in positions if p[2] in val_game_ids]
    
    print(f"Training positions: {len(train_positions)}, Validation positions: {len(val_positions)}")
    
    # --- Convert to tensors and save ---
    def save_positions(pos_list, filename):
        boards = torch.stack([torch.FloatTensor(p[0]) for p in pos_list])
        moves = torch.LongTensor([p[1] for p in pos_list])
        torch.save({
            'boards': boards,
            'moves': moves,
            'num_positions': len(pos_list)
        }, output_dir / filename)
        print(f"✓ Saved {filename}, size: {(output_dir / filename).stat().st_size / 1e6:.1f} MB")
    
    save_positions(train_positions, 'train_data.pt')
    save_positions(val_positions, 'val_data.pt')
    
    # --- Save vocabulary ---
    vocab_path = output_dir / 'move_vocab.json'
    with open(vocab_path, 'w') as f:
        json.dump({'move_to_idx': move_to_idx, 'num_moves': len(move_to_idx)}, f)
    print(f"✓ Saved vocabulary: {vocab_path}, size: {len(move_to_idx)} moves")

# TESTING
def count_positions(pgn_path, opening_sample_rate=0.2):
    total_positions = 0
    game_count = 0

    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            board = game.board()
            move_num = 0
            for move in game.mainline_moves():
                move_num += 1

                # Optionally downsample first 5 opening moves
                if move_num <= 5 and random.random() > opening_sample_rate:
                    board.push(move)
                    continue

                total_positions += 1
                board.push(move)

            game_count += 1

    print(f"Processed {game_count} games, found {total_positions} positions")
    return total_positions

if __name__ == '__main__':
    vocab = build_vocabulary()
    positions = process_pgn('data/raw/dataset.pgn', vocab, 2500000)
    save_data(positions, vocab)
    