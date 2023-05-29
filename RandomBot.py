# -*- coding: utf-8 -*-
"""
Created on Tue May 23 07:07:17 2023

@author: giuse
"""

import random
import pandas as pd
import os
from time import sleep
import random
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget

import chess
import chess.svg

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(80, 80, 880, 880)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(8, 8, 800, 800)

        self.chessboard = chess.Board()

        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)
# =============================================================================
# random bot
# =============================================================================
def RandomBot(board):
    move_count = board.legal_moves.count()-1
    random_move = random.randint(0, move_count)
    bot_move = str(list(board.legal_moves)[random_move])
    return bot_move

#define bot Moves

def BOT_MOVE(board, player_move):
# =============================================================================
#         MODEL goes here: in the selection of a move
#         1. RANDOM
#         2. Basic Probability
# =============================================================================
  
        bot_move = RandomBot(board)
        #bot_move = PepeBot.PEPE_BOT_PROB(df, board, player_move)
        
        
        if bot_move == '':
            return bot_move, board
        #if not pawn promotion
        if not pawn_promotion(board, bot_move):
            #print(bot_move)
            board.push_san(bot_move)
            
        else:
            
            #bot chooses promotion piece
            #1. always pick Queen
            board.push_san(bot_move +'q')
            #2. random choice
            #3. Pick a piece that makes sense
            
# =============================================================================
#         #display board
#         chessboardSvg = chess.svg.board(board,flipped=True).encode("UTF-8")
#         window.widgetSvg.load(chessboardSvg)
#         app.processEvents()
#         sleep(random.uniform(1, 4)) #time before bot plays
# =============================================================================
        
        #flip the board for opponent
        chessboardSvg = chess.svg.board(board,flipped=True).encode("UTF-8")
        window.widgetSvg.load(chessboardSvg)
        app.processEvents()
        
        return bot_move
    
    
#define user plays

def USER_MOVE(board):
        #if it's checkmate:
        if board.is_checkmate():
            # Determine the winner based on the last move played
            last_move = board.peek()  # Get the last move without modifying the board
            winner = "White" if board.turn == chess.BLACK else "Black"
            print("Winner:", winner)
            
            player_move = '' #just to output same thing
            return player_move, winner
        else: #if not checkmate
            legal_moves = [move.uci() for move in board.legal_moves]
            while True:
                try:
                    player_move = input("Your move:")  
                    #if not pawn promotion
                    if not pawn_promotion(board, player_move):
                        board.push_san(player_move)
                    else:
                        #user chooses promotion piece
                        promotion_piece = input("Piece to promote to (q, r, k, b):")
                        board.push_san(player_move + promotion_piece)
                    break
                except ValueError or player_move not in legal_moves:
                    print("Illegal move. Try again.")
        
# =============================================================================
#         Display Board
# =============================================================================
            
            chessboardSvg = chess.svg.board(board,flipped=True).encode("UTF-8")
            window.widgetSvg.load(chessboardSvg)
            window.show()
            app.processEvents()
            sleep(2)

        return player_move, board
    
# =============================================================================
#   check if it's pawn promotion
# =============================================================================
def pawn_promotion(board, move):
    
    print(move)
    #get piece type from location
    square = chess.parse_square(move[:2])
    piece = board.piece_at(square)
    if len(move) != 4 or type(piece) is type(None):  # If not a valid move
        print("Invalid move:", move)
        return False
    #all_squares = [chess.square_name(square) for square in range(64)]
    else:
        
        
        #get file of first and second move
        start_rank = int(move[1])
        end_rank = int(move[3])
        #print(move)
        #print(piece)
        if piece.piece_type == chess.PAWN and ((start_rank == 2 and end_rank == 1) or (start_rank == 7 and end_rank == 8)):
            #this is a promotion, tag it as such:
            promotion = True
        else:
            promotion = False
        
    return promotion

# =============================================================================
# START THE GAME
# =============================================================================

if __name__ == "__main__":
    app = QApplication([])

    #set up board
    board = chess.Board()
    #print(board)
    i=0
    #initialize 0th move for bot
    #IF BOT is White, set this to empty, otherwise no need to define

    player_move = ''
    bot_move = 'not empty'
    window = MainWindow()
    
    chessboardSvg = chess.svg.board(board, flipped=True).encode("UTF-8")
    window.widgetSvg.load(chessboardSvg)
    window.show()
    app.processEvents()
    
    #Train AI based on data:
    #model, legal_features, legal_y = PepeBot.PEPE_AI_Train(X, y, board)
    
    while not ValueError or player_move == '' or bot_move != '':
        i=i+1
        print("Turn number " + str(i))
        
        
        # =============================================================================
        #  WHITE MOVES
        # =============================================================================
        bot_move = BOT_MOVE(board, player_move)
        #bot_move = AI_BOT(model, X, y, board)
        
        # =============================================================================
        #  BLACK MOVES
        # =============================================================================
        player_move, board = USER_MOVE(board)
        
    #app.exec()
    app.quit()
