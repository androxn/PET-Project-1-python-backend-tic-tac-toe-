class Board:
    """Класс, который описывает игровое поле."""
    field_size = 3
    # Инициализировать игровое поле - список списков с пробелами.
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]

    # Метод, который обрабатывает ходы игроков.
    def make_move(self, row, col, player):
        self.board[row][col] = player

    # Метод, который отрисовывает игровое поле.
    def display(self):
        for row in self.board:
            print('|'.join(row))
            print('-' * 5)
    def __str__(self):
        return (
            'Объект игрового поля размером '
            f'{self.field_size}x{self.field_size}'
        )

    def check_win(self, player):
        """Проверяет, победил ли игрок."""
        # Проверка строк
        for i in range(3):
            if all([self.board[i][j] == player for j in range(3)]):
                return True

        # Проверка столбцов
        for i in range(3):
            if all([self.board[j][i] == player for j in range(3)]):
                return True

        # Проверка главной диагонали
        if all([self.board[i][i] == player for i in range(3)]):
            return True

        # Проверка побочной диагонали
        if all([self.board[i][2 - i] == player for i in range(3)]):
            return True

        return False

    def is_draw(self):
        """Проверяет, закончилась ли игра ничьей."""
        for row in self.board:
            if ' ' in row:
                return False  # Есть пустые клетки - игра продолжается
        return True  # Все клетки заполнены - ничья

    def is_move_possible(self, row, col):
        """Проверяет, можно ли сделать ход в указанную ячейку."""
        if row < 0 or row >= self.field_size or col < 0 or col >= self.field_size:
            return False
        return self.board[row][col] == ' '
