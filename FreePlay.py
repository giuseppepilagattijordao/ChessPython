# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:05:37 2023

@author: giuse
"""

import random
import pandas as pd

import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import chess
import chess.svg
from time import sleep
import random
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget

import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from stockfish import Stockfish

import numpy as np
import tensorflow as tf
import chess
import chess.engine #to use stockfish evaluation


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(80, 80, 880, 880)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(8, 8, 800, 800)

        self.chessboard = chess.Board()

        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)
        
        
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
                    if board.turn == chess.BLACK:
                        player_move = input("Black move: ") 
                    else:
                        player_move = input("White move: ")  
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
# START GAME: Human VS Human
# =============================================================================

if __name__ == "__main__":
    app = QApplication([])

    #set up board
    board = chess.Board()
    #print(board)
    i=0
    #initialize 0th move for bot
    #IF BOT is White, set this to empty, otherwise no need to define

    player_move1 = ''
    player_move2 = ''
    
    window = MainWindow()
    
    chessboardSvg = chess.svg.board(board, flipped=True).encode("UTF-8")
    window.widgetSvg.load(chessboardSvg)
    window.show()
    app.processEvents()
    
    
    while not ValueError or player_move1 == '' or player_move2 != '':
        i=i+1
        print("Turn number " + str(i))
        # =============================================================================
        #  WHITE MOVES
        # =============================================================================
        #flip the board for white
            
        player_move1, board = USER_MOVE(board)
        
        # =============================================================================
        #  BLACK MOVES
        # =============================================================================

        player_move2, board = USER_MOVE(board)
        
    #app.exec()
    app.quit()




