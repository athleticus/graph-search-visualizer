import json
import tkinter as tk
import time
import copy
from ee import EventEmitter
from PIL import ImageGrab
from PIL import Image, ImageDraw
import os

WIDTH = 1000
HEIGHT = 800

CELL_WIDTH = 50
CELL_HEIGHT = 50

X_OFFSET = 2
Y_OFFSET = 2

PATH_WIDTH = 120
PATH_HEIGHT = 120

CURRENT_NODE_FILL = 'gold'
NONCURRENT_NODE_FILL = 'alice blue'
FINISHED_CELL_FILL = 'firebrick'

FINISH_CELL_FILL = 'gold'
FINISH_CELL_FILL = '#F2EFD3'
CELL_LABEL_COLOUR = '#808080'
UNVISITED_FILL = '#ffffff'
NEIGHBOUR_CELL_FILL = 'green yellow'

START_CELL_FILL = '#D3E8F2'
STACK_CELL_FILL = '#F2D3D5'
VISITED_CELL_FILL = 'indian red'
REVISITED_CELL_FILL = 'indian red'

NORTH = 'n'
EAST = 'e'
SOUTH = 's'
WEST = 'w'

DIRECTIONS = [NORTH, EAST, SOUTH, WEST]

DIRECTION_DELTAS = {
    NORTH: (-1, 0),
    SOUTH: (1, 0),
    EAST: (0, 1),
    WEST: (0, -1)
}

DFS_WHITE = None
DFS_GRAY = False
DFS_BLACK = True


class Maze(EventEmitter):
    def __init__(self, rows, columns, walls=None, directed=False):
        super().__init__()

        self._rows = rows
        self._columns = columns

        self._cells = [[None for j in range(columns)] for i in range(rows)]
        self._walls = set()

        self._directed = directed

        if walls:
            for from_row, from_col, to_row, to_col in walls:
                from_pos = from_row, from_col
                to_pos = to_row, to_col

                self._walls.add((from_pos, to_pos))

                if not directed:
                    self._walls.add((to_pos, from_pos))

    def get_rows(self):
        return self._rows

    def get_columns(self):
        return self._columns

    def get_walls(self):
        return list(self._walls)

    def set_positions(self, start, finishes):
        self._start = tuple(start)
        self._finishes = set([tuple(p) for p in finishes])

    def is_position_finish(self, pos):
        return pos in self._finishes

    def is_position_start(self, pos):
        return pos == self._start

    def get_positions(self):
        return self._start, self._finishes

    def is_directed(self):
        return self._directed

    def __getitem__(self, item):
        row, column = item
        return self._cells[row][column]

    def __setitem__(self, key, value):
        row, column = key
        self._cells[row][column] = value

    def __iter__(self):
        for row, row_data in enumerate(self._cells):
            for column, value in enumerate(row_data):
                yield row, column

    def items(self):
        for pos in iter(self):
            yield pos, self[pos]

    def can_move(self, from_pos, to_pos):
        if not self.is_position_valid(from_pos) or not self.is_position_valid(
                to_pos):
            return False

        return (from_pos, to_pos) not in self._walls

    def is_position_valid(self, pos):
        row, column = pos

        return 0 <= row < self._rows and 0 <= column < self._columns

    def get_neighbours(self, pos):
        row, column = pos
        for dir in DIRECTIONS:
            dr, dc = DIRECTION_DELTAS[dir]
            new_pos = row + dr, column + dc

            if self.is_position_valid(new_pos) and self.can_move(pos, new_pos):
                yield new_pos

    def clone(self):
        return copy.deepcopy(self)

    @classmethod
    def load_data(cls, data):
        m = cls(data['rows'], data['columns'], data.get('walls'),
                directed=data.get('directed'))

        m.set_positions(data['start'], data['finishes'])

        return m

    @classmethod
    def load_file(cls, filename):
        with open(filename, 'r') as f:
            data = json.loads(f.read())
            return cls.load_data(data)

    def dfs(self, stop_at_finish=False, stop_if_revisiting=False,
            revisit_allowed=False):
        # Find finish from start using dfs
        self.emit('dfs', 'start')
        yield
        start, finishes = self.get_positions()
        finish = next(iter(finishes))

        S = []
        S.append(start)
        self[start] = DFS_GRAY
        self.emit('dfs', 'push_start', start)
        yield

        while len(S):
            v = S[-1]
            self.emit('dfs', 'current', v)
            yield

            u = None
            for neighbour in self.get_neighbours(v):

                if self[neighbour] != DFS_WHITE:
                    self.emit('dfs', 'revisit', neighbour)
                    # yield
                    if self[neighbour] == DFS_GRAY:
                        if stop_if_revisiting:
                            self.emit('dfs', 'cycle', neighbour)
                            return

                    continue

                self.emit('dfs', 'neighbour', neighbour)
                # yield
                u = neighbour
                break

            if u:
                self.emit('dfs', 'push', u)
                yield
                S.append(u)
                self[u] = DFS_GRAY

                if stop_at_finish and u in finishes:
                    self.emit('dfs', 'goal', u)
                    return
            else:
                self.emit('dfs', 'pop', v)
                yield
                S.pop(-1)
                self[v] = DFS_BLACK

        self.emit('dfs', 'failed')

    def dfs_old(self, stop_at_finish=False, stop_if_revisiting=False,
                revisit_allowed=False):
        # Find finish from start using dfs
        self.emit('dfs', 'start')
        start, finishes = self.get_positions()
        finish = next(iter(finishes))

        S = []
        S.append(start)
        self.emit('dfs', 'push_start', start)
        yield

        while len(S):
            v = S.pop()
            self.emit('dfs', 'pop', v)
            yield
            if self[v] is None:
                self[v] = True
                self.emit('dfs', 'first_visit', v)
                yield
                for neighbour in reversed(list(self.get_neighbours(v))):
                    if stop_at_finish and neighbour in finishes:
                        self.emit('dfs', 'goal', neighbour)
                        return

                    if self[neighbour]:
                        if not revisit_allowed:
                            continue

                        self.emit('dfs', 'second_visit', neighbour)
                        if stop_if_revisiting:
                            return
                        else:
                            yield
                    S.append(neighbour)
                    self.emit('dfs', 'push', neighbour)
                    yield
                self.emit('dfs', 'push_finish', v)
                yield

        self.emit('dfs', 'failed')


DFS_MEMORY = """
Stack: {stack}
Visited: {visited}
Current: {current}
""".strip()


class MazeMemory(object):
    def __init__(self, canvas, maze, bounds, font=None, *args, **kwargs):
        self._canvas = canvas

        self._maze = maze

        self._font = font

        self._top_left, self._bottom_right = bounds

        self._stack = []
        self._neighbours = []
        self._visited = []
        self._current = None

        self._item = None

        self.setup_listeners()

    def _handle_dfs(self, type, pos=None, pos2=None):
        if type == 'start':
            pass
        elif type == 'push_start':
            self._stack.append(pos)
        elif type == 'current':
            self._current = pos
        elif type == 'goal':
            self._neighbours.append(pos)
        elif type == 'neighbour':
            pass
        elif type == 'revisit':
            pass
        elif type == 'push':
            self._stack.append(pos)
        elif type == 'pop':
            self._stack.pop(-1)
            self._visited.append(pos)
        elif type == 'failed':
            pass

        # 'start'
        # 'push_start', start
        # 'current', v
        # 'goal',neighbour
        # 'neighbour', neighbour
        # 'revisit', neighbour
        # 'push', v, u
        # 'pop', v
        # 'failed'

        self.draw()

    def setup_listeners(self):
        self._maze.on('dfs', self._handle_dfs)

    def draw(self):
        if not self._item:
            self._item = self._canvas.create_text(self._top_left, anchor=tk.NW,
                                                  font=self._font)

        text = DFS_MEMORY.format(**{
            'current': self._current,
            'stack': ", ".join(
                [str(v) for v in self._stack]) if self._stack else '',
            'neighbours': ", ".join(
                [str(v) for v in self._neighbours]) if self._neighbours else '',
            'visited': ", ".join(
                [str(v) for v in self._visited]) if self._visited else ''
        })

        self._canvas.itemconfigure(self._item, text=text)


class Data:
    def __init__(self):
        self._history = []

    def add(self, type, data):
        self._history.append((type, data))

    def find_last(self, type=None):
        if type is None:
            return self._history[-1]

        for this_type, this_data in reversed(self._history):
            if this_type == type:
                return this_type, this_data


class MazeView(tk.Canvas):
    def __init__(self, master, maze, width=WIDTH, height=HEIGHT,
                 cell_width=CELL_WIDTH, cell_height=CELL_HEIGHT,
                 stats_height=110):
        self._master = master
        super().__init__(master, width=width, height=(height + stats_height))

        self._maze = maze
        self._width = width
        self._height = height
        self._cell_width = cell_width
        self._cell_height = cell_height
        self._cell_padding_x = width / maze.get_columns() - self._cell_width
        self._cell_padding_y = height / maze.get_rows() - self._cell_height

        self._visited_bg = maze.clone()
        self._node_bg = maze.clone()

        self._memory = MazeMemory(self, self._maze, (
        (0 + 8, height + 8), (width, height + stats_height)),
                                  font=('Helvetica', '28'))

        self.setup_listeners()

        self._im = Image.new("RGB", (width, height + stats_height))
        self._draw = ImageDraw.Draw(self._im)

        self.draw_grid()

    def setup_listeners(self):
        self._maze.on('dfs', self._handle_dfs)

    def _handle_dfs(self, type, pos=None, pos2=None):
        print(type, pos, pos2)

        if type == 'start':
            data = self._data = Data()
            data.prev_neighbours = []
        elif type == 'push_start':
            print("Starting at {}".format(pos))
            self.itemconfigure(self._visited_bg[pos], fill=STACK_CELL_FILL)
        elif type == 'current':
            print('Popped {}'.format(pos))
            fill = CURRENT_NODE_FILL  # if pos != self._maze.get_positions()[0] else START_CELL_FILL
            self.itemconfigure(self._node_bg[pos], fill=CURRENT_NODE_FILL)
            last = self._data.find_last('current')
            if last:
                last_type, last_pos = last
                self.itemconfigure(self._node_bg[last_pos],
                                   fill=NONCURRENT_NODE_FILL)

        elif type == 'goal':
            self.itemconfigure(self._node_bg[pos],
                               fill=CURRENT_NODE_FILL)
            self.itemconfigure(self._visited_bg[pos],
                               fill=STACK_CELL_FILL)

            last = self._data.find_last('current')
            if last:
                last_type, last_pos = last
                self.itemconfigure(self._node_bg[last_pos],
                                   fill=NONCURRENT_NODE_FILL)
        elif type == 'neighbour':
            pass
        elif type == 'revisit':
            pass
        elif type == 'cycle':
            print("Found cycle {}".format(pos))
            self.itemconfigure(self._node_bg[pos], fill=REVISITED_CELL_FILL)
        elif type == 'push':
            print("Pushing {}".format(pos))
            self.itemconfigure(self._visited_bg[pos], fill=STACK_CELL_FILL)
        elif type == 'pop':
            self.itemconfigure(self._visited_bg[pos], fill=VISITED_CELL_FILL)
        elif type == 'failed':
            pass

        self._data.add(type, pos)

    def draw_grid(self, graph_view=True):
        self.delete(tk.ALL)

        # Draw background
        for row in range(self._maze.get_rows()):
            for column in range(self._maze.get_columns()):
                self.draw_background((row, column))

        # self._im.show()
        # return

        # Draw boundaries
        for row in range(self._maze.get_rows() + 1):
            _, y = self.get_cell_pos((row, 0), centre=False)
            self.create_line(0, y, self._width, y, dash=(5, 5), fill='gray')
        for column in range(self._maze.get_columns() + 1):
            x, _ = self.get_cell_pos((0, column), centre=False)
            self.create_line(x, 0, x, self._height, dash=(5, 5), fill='gray')

        # Draw walls
        for row in range(self._maze.get_rows()):
            for column in range(self._maze.get_columns()):

                for dir in (EAST, SOUTH):
                    dr, dc = DIRECTION_DELTAS[dir]
                    from_pos = row, column
                    to_pos = (row + dr, column + dc)

                    if not self._maze.can_move(from_pos,
                                               to_pos) and not self._maze.can_move(
                            to_pos, from_pos):
                        self.draw_wall(from_pos, to_pos)

        if not graph_view:
            return

        # Draw cells
        for row in range(self._maze.get_rows()):
            for column in range(self._maze.get_columns()):
                self.draw_cell((row, column))

        # Draw paths
        for row in range(self._maze.get_rows()):
            for column in range(self._maze.get_columns()):

                for dir in (EAST, SOUTH):
                    dr, dc = DIRECTION_DELTAS[dir]
                    from_pos = row, column
                    to_pos = (row + dr, column + dc)

                    if not (self._maze.is_position_valid(
                            from_pos) and self._maze.is_position_valid(to_pos)):
                        continue

                    self.draw_paths((row, column), (row + dr, column + dc))

    def draw_background(self, pos):
        x, y = self.get_cell_pos(pos)

        start, finishes = self._maze.get_positions()

        if pos == start:
            special_fill = START_CELL_FILL
        elif pos in finishes:
            special_fill = FINISH_CELL_FILL
        else:
            special_fill = UNVISITED_FILL

        x1, y1 = self.get_cell_pos(pos, centre=False)
        x2, y2 = self.get_cell_pos((pos[0] + 1, pos[1] + 1), centre=False)

        self._visited_bg[pos] = self.create_rectangle(x1, y1, x2, y2,
                                                      fill=special_fill,
                                                      outline='')

        res = self._draw.rectangle((x1, y1, x2, y2), fill=special_fill,
                                   outline=None)

    def get_cell_pos(self, rc_pos, centre=True):
        row, col = rc_pos
        offset = 0.5 if centre else 0
        x = (col + offset) * (
        self._cell_width + self._cell_padding_x) + X_OFFSET
        y = (row + offset) * (
        self._cell_height + self._cell_padding_y) + Y_OFFSET

        return x, y

    def draw_cell(self, pos):
        x, y = self.get_cell_pos(pos)

        start, finishes = self._maze.get_positions()

        if pos == start:
            text = 'S'
        elif pos in finishes:
            text = 'X'
        else:
            text = ''

        fill = VISITED_CELL_FILL if self._maze[pos] else NONCURRENT_NODE_FILL

        self._node_bg[pos] = self.create_oval(x - self._cell_width // 2,
                                              y - self._cell_height // 2,
                                              x + self._cell_width // 2,
                                              y + self._cell_height // 2,
                                              fill=fill)

        if text:
            self.create_text(x, y, text=text, font=("Helvetica", 28))

        # Draw label
        x_end, y_end = self.get_cell_pos((pos[0] + 1, pos[1] + 1), centre=False)
        text_x, text_y = (x + x_end) // 2, (y + y_end) // 2
        self.create_text(text_x, text_y, text=str(pos), font=("Helvetica", 28),
                         fill=CELL_LABEL_COLOUR)

    def draw_paths(self, from_pos, to_pos):
        can_from_to = self._maze.can_move(from_pos, to_pos)
        can_to_from = self._maze.can_move(to_pos, from_pos)

        if can_from_to and can_to_from:
            arrow = tk.BOTH
        elif can_from_to:
            arrow = tk.LAST
        elif can_to_from:
            arrow = tk.FIRST
        else:
            arrow = None

        x1, y1 = self.get_cell_pos(from_pos)
        x2, y2 = self.get_cell_pos(to_pos)

        dx, dy = (x2 - x1) // 2, (y2 - y1) // 2
        x_mid, y_mid = x1 + dx, y1 + dy

        if dx:
            x1 = x_mid - PATH_WIDTH // 2
            x2 = x_mid + PATH_WIDTH // 2
        if dy:
            y1 = y_mid - PATH_HEIGHT // 2
            y2 = y_mid + PATH_HEIGHT // 2

        if arrow is not None:
            # no arrow heads for undirected
            # arrow if self._maze.is_directed() else None
            self.create_line(x1, y1, x2, y2, arrow=arrow)

    def draw_wall(self, from_pos, to_pos):
        from_row, from_column = from_pos
        to_row, to_column = to_pos

        is_horizontal = from_row != to_row

        if is_horizontal:
            r1 = r2 = to_row
            c1, c2 = to_column, to_column + 1
        else:
            r1, r2 = to_row, to_row + 1
            c1 = c2 = to_column

        x1, y1 = self.get_cell_pos((r1, c1), centre=False)
        x2, y2 = self.get_cell_pos((r2, c2), centre=False)

        self.create_rectangle(x1, y1, x2, y2)


class MazeApp(object):
    ANIMATION_DELAY = 1500

    def __init__(self, master):
        self._master = master

        m = Maze.load_file('dag1.json')
        self._maze = m

        print(m.get_walls())

        self._view = MazeView(master, m)
        self._view.pack(side=tk.TOP)

        self.setup_menus()

        #     self._view.bind("<Configure>", self._resize)
        #
        # def _resize(self, ev):
        #     self.run_dfs_goal_1()

    def setup_menus(self):
        menubar = tk.Menu(self._master)

        # create a pulldown menu, and add it to the menu bar
        demo_menu = tk.Menu(menubar, tearoff=0)
        demo_menu.add_command(label="DFS -> Goal (Undirected)",
                              command=self.run_dfs_goal_1)
        demo_menu.add_command(label="DFS -> Goal (Directed A -> B)",
                              command=self.run_dfs_goal_2a)
        demo_menu.add_command(label="DFS -> Goal (Directed B -> A)",
                              command=self.run_dfs_goal_2b)
        demo_menu.add_command(label="DFS -> Detect DAG",
                              command=self.run_dfs_dag_1)
        demo_menu.add_command(label="Intro", command=self.run_intro)
        demo_menu.add_separator()
        demo_menu.add_command(label="Exit", command=self._master.quit)
        menubar.add_cascade(label="Demos", menu=demo_menu)

        self._master.config(menu=menubar)

    def run_dfs_goal_1(self):
        master = self._master

        self._view.pack_forget()

        m = Maze.load_file('undirected1.json')
        self._maze = m

        self._view = MazeView(master, m)
        self._view.pack(side=tk.TOP)

        moves = self._maze.dfs(stop_at_finish=True, stop_if_revisiting=False,
                               revisit_allowed=True)

        filename = "images/find-goal-undirected/find-goal-undirected_{}.png"

        moves = self._maze.dfs(stop_at_finish=True, stop_if_revisiting=False,
                               revisit_allowed=True)

        self._animate_moves(moves, filename=filename)

    def run_dfs_goal_2a(self):
        master = self._master

        self._view.pack_forget()

        m = Maze.load_file('directed1a.json')
        self._maze = m

        self._view = MazeView(master, m)
        self._view.pack(side=tk.TOP)

        filename = "images/find-goal-directed-a-b/find-goal-directed-a-b_{}.png"

        moves = self._maze.dfs(stop_at_finish=True, stop_if_revisiting=False,
                               revisit_allowed=True)

        self._animate_moves(moves, filename=filename)

    def run_dfs_goal_2b(self):
        master = self._master

        self._view.pack_forget()

        m = Maze.load_file('directed1b.json')
        self._maze = m

        self._view = MazeView(master, m)
        self._view.pack(side=tk.TOP)

        capture = True

        if capture:
            dim, x, y = master.geometry().split('+')
            width, height = dim.split('x')

        filename = "images/find-goal-directed-b-a/find-goal-directed-b-a_{}.png"

        moves = self._maze.dfs(stop_at_finish=True, stop_if_revisiting=False,
                               revisit_allowed=True)

        self._animate_moves(moves, filename=filename)

    def run_dfs_dag_1(self):
        master = self._master

        self._view.pack_forget()

        m = Maze.load_file('dag1.json')
        self._maze = m

        self._view = MazeView(master, m)
        self._view.pack(side=tk.TOP)

        filename = "images/detect-cycle/detect-cycle_{}.png"

        moves = self._maze.dfs(stop_at_finish=True, stop_if_revisiting=True,
                               revisit_allowed=True)
        self._animate_moves(moves, filename)

    def run_intro(self):
        master = self._master

        self._view.pack_forget()

        m = Maze.load_file('undirected1.json')
        self._maze = m

        self._view = MazeView(master, m)
        self._view.pack(side=tk.TOP)

        filename = "images/intro/intro_{}.png"

        def _intro():
            yield self._view.draw_grid(graph_view=False)
            yield self._view.draw_grid(graph_view=True)

        moves = _intro()

        self._animate_moves(moves, filename=filename)

    def _animate_moves(self, moves, filename=None):

        class Counter:
            i = 0

        if filename:
            dim, x, y = self._master.geometry().split('+')
            width, height = dim.split('x')

            os.makedirs(os.path.dirname(filename), exist_ok=True)

            def cap():
                self._view.update()
                ImageGrab.grab((int(x), int(y) + 20, int(x) + int(width),
                                int(y) + 20 + int(height) - 120)).save(
                    filename.format(Counter.i))

        def _run_dfs():
            try:
                next(moves)
            except StopIteration:
                if filename:
                    cap()

                return

            Counter.i += 1

            if filename:
                cap()

            self._master.after(self.ANIMATION_DELAY, _run_dfs)

        self._master.after(self.ANIMATION_DELAY, _run_dfs)


def main():
    root = tk.Tk()
    MazeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
