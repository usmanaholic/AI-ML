import tkinter as tk
from tkinter import messagebox
import numpy as np
from collections import defaultdict

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_winner = None

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.check_winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def check_winner(self, square, letter):
        row_ind = square // 3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all([s == letter for s in row]):
            return True
        
        col_ind = square % 3
        col = [self.board[col_ind+i*3] for i in range(3)]
        if all([s == letter for s in col]):
            return True
        
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([s == letter for s in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([s == letter for s in diagonal2]):
                return True
        return False

class QLearningAgent:
    def __init__(self, alpha=0.5, epsilon=0.1):
        self.q_table = defaultdict(dict)
        self.alpha = alpha
        self.epsilon = epsilon
        self.history = []

    def get_state(self, board):
        return tuple(board)

    def choose_action(self, state, available_actions):
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        else:
            q_values = [self.q_table[state].get(a, 0) for a in available_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
            return np.random.choice(best_actions)

    def update_q_table(self, reward):
        for (state, action) in self.history:
            current_q = self.q_table[state].get(action, 0)
            new_q = current_q + self.alpha * (reward - current_q)
            self.q_table[state][action] = new_q
        self.history = []

class TicTacToeGUI:
    def __init__(self, agent):
        self.agent = agent
        self.game = TicTacToe()
        self.human_symbol = ''
        self.ai_symbol = ''
        
        # Initial symbol selection window
        self.setup_window = tk.Tk()
        self.setup_window.title("Choose Symbol")
        tk.Label(self.setup_window, text="Choose your symbol:", font=('Arial', 14)).pack(pady=10)
        tk.Button(self.setup_window, text='X', font=('Arial', 20), width=4, 
                 command=lambda: self.start_game('X')).pack(side=tk.LEFT, padx=20)
        tk.Button(self.setup_window, text='O', font=('Arial', 20), width=4,
                 command=lambda: self.start_game('O')).pack(side=tk.RIGHT, padx=20)
        self.setup_window.mainloop()

    def start_game(self, symbol):
        self.setup_window.destroy()
        self.human_symbol = symbol
        self.ai_symbol = 'O' if symbol == 'X' else 'X'
        
        # Main game window
        self.window = tk.Tk()
        self.window.title(f"Tic Tac Toe - You: {self.human_symbol} vs AI: {self.ai_symbol}")
        
        self.buttons = []
        for i in range(9):
            btn = tk.Button(self.window, text='', font=('Arial', 40), width=4, height=2,
                          command=lambda i=i: self.human_move(i))
            btn.grid(row=i//3, column=i%3)
            self.buttons.append(btn)
        
        self.status_label = tk.Label(self.window, text="", font=('Arial', 14))
        self.status_label.grid(row=3, columnspan=3)
        
        # Determine who starts first
        if self.human_symbol == 'X':
            self.status_label.config(text="Your turn (X)")
            for btn in self.buttons:
                btn.config(state='normal')
        else:
            self.status_label.config(text="AI's turn")
            self.ai_turn()
        
        self.window.mainloop()

    def ai_turn(self):
        self.status_label.config(text="AI is thinking...")
        self.window.update()
        
        # Create mirrored state if AI is playing O
        if self.ai_symbol == 'O':
            mirrored_board = ['X' if c == 'O' else 'O' if c == 'X' else ' ' 
                            for c in self.game.board]
            state = self.agent.get_state(mirrored_board)
        else:
            state = self.agent.get_state(self.game.board)
            
        available_actions = self.game.available_moves()
        
        if available_actions:
            action = self.agent.choose_action(state, available_actions)
            self.game.make_move(action, self.ai_symbol)
            self.buttons[action].config(text=self.ai_symbol, 
                                      fg='red' if self.ai_symbol == 'X' else 'green',
                                      state='disabled')
            
            if self.game.current_winner:
                self.game_over("AI wins!")
                return
            elif not self.game.available_moves():
                self.game_over("It's a tie!")
                return
            
            self.status_label.config(text=f"Your turn ({self.human_symbol})")
            for btn in self.buttons:
                if btn['text'] == '':
                    btn.config(state='normal')

    def human_move(self, square):
        if self.game.make_move(square, self.human_symbol):
            self.buttons[square].config(text=self.human_symbol, 
                                      fg='blue', state='disabled')
            
            if self.game.current_winner:
                self.game_over("You win!")
                return
            elif not self.game.available_moves():
                self.game_over("It's a tie!")
                return
            
            for btn in self.buttons:
                btn.config(state='disabled')
            self.window.after(500, self.ai_turn)

    def game_over(self, message):
        for btn in self.buttons:
            btn.config(state='disabled')
        answer = messagebox.askyesno("Game Over", f"{message}\nPlay again?")
        if answer:
            self.reset_game()
        else:
            self.window.destroy()

    def reset_game(self):
        self.game = TicTacToe()
        for btn in self.buttons:
            btn.config(text='', state='normal')
        
        if self.human_symbol == 'X':
            self.status_label.config(text="Your turn (X)")
            for btn in self.buttons:
                btn.config(state='normal')
        else:
            for btn in self.buttons:
                btn.config(state='disabled')
            self.status_label.config(text="AI's turn")
            self.ai_turn()

def train(agent, episodes=10000):
    for _ in range(episodes):
        game = TicTacToe()
        agent.history = []
        current_player = 'X'
        reward = 0
        
        while game.available_moves():
            if current_player == 'X':
                state = agent.get_state(game.board)
                action = agent.choose_action(state, game.available_moves())
                game.make_move(action, 'X')
                agent.history.append((state, action))
                
                if game.current_winner:
                    reward = 1
                    break
                current_player = 'O'
            else:
                action = np.random.choice(game.available_moves())
                game.make_move(action, 'O')
                if game.current_winner:
                    reward = -1
                    break
                current_player = 'X'
        
        agent.update_q_table(reward)
        agent.epsilon = max(0.01, agent.epsilon * 0.999)

if __name__ == "__main__":
    agent = QLearningAgent()
    train(agent, episodes=10000)
    TicTacToeGUI(agent)