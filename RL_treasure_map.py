
# ZMIENIĆ SCIEŻKE DO ZAPISU OBRAZKOW

import numpy as np
import copy
import itertools
import random
import matplotlib.pyplot as plt
import time


def create_environment(TRASURE_MAP_SIZE_X, TRASURE_MAP_SIZE_Y, print_info=False):
    # Inicjujemy mape skarbow
    treasure_map = np.random.choice([0, 1], p=[0.65, 0.35], size=(TRASURE_MAP_SIZE_X, TRASURE_MAP_SIZE_Y))

    # Inicjujemy punkt startowy
    start_point = np.random.randint(0, 9, size=2)
    if print_info:
        print('START:',start_point)

    # Inicjujemy miejsce skarbu - punkt koncowy
    end_point = np.random.randint(0, 9, size=2)
    if print_info:
        print('END',end_point)

    # Zaznaczamy '-1' mijesce gdzie jst ukryty skarb
    treasure_map[end_point[0], end_point[1]] = -1
    if print_info:
        print('MAP:',treasure_map)
    
    return treasure_map, start_point, end_point

def move_agent(action, current_point):
    next_point =  copy.deepcopy(current_point)
    if action == 'N':
        next_point[0] = current_point[0] - 1
    if action == 'S':
        next_point[0] = current_point[0] + 1
    if action == 'W':
        next_point[1] = current_point[1] - 1
    if action == 'E':
        next_point[1] = current_point[1] + 1
    return next_point

def is_no_the_map(next_point, treasure_map):
    'Funkcja sprawdza czy dany punkt nastepny (next_x, next_y) znajduje sie na okreslonej mapie'
    if (next_point[1] < 0) or (next_point[1] >= treasure_map.shape[1]) or (next_point[0] < 0) or (next_point[0] >= treasure_map.shape[0]):
        return False
    else:
        return True

def is_treasure_found(current_point, end_point):
    'Funkcja sprawdza czy znaleziono skarb'
    if (current_point[1] == end_point[1]) and (current_point[0] == end_point[0]):
        return True
    else:
        False

def reward_value(next_point, hist_path, treasure_map):
    'Funkcja zwracajaca nagrode'
    if treasure_map[next_point[0], next_point[1]] == 1: # jesli natrafimy na przeszkode
        reward_value = -50
    if treasure_map[next_point[0], next_point[1]] == 0: # jesli nie natrafimy na przeszkode
        reward_value = 1
    if treasure_map[next_point[0], next_point[1]] == -1: # jesli natrafimy skarb
        reward_value = 100
    
    if tuple(next_point) in hist_path:
        reward_value = reward_value - 10
    return reward_value

def get_current_state(current_point, treasure_map):
    PADDING = 2
    def pad_with(array, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', -9)
        array[:pad_width[0]] = pad_value
        array[-pad_width[1]:] = pad_value

    treasure_map_temp = np.pad(treasure_map, PADDING, pad_with)
    current_point_temp = current_point + PADDING
    current_state = treasure_map_temp[(current_point_temp[0]-1):(current_point_temp[0]+2), (current_point_temp[1]-1):(current_point_temp[1]+2)].flatten()
    current_state = np.delete(current_state, 4)
    return tuple(current_state)

def reverse_direction(direction):
    if direction == 'N':
        new_direction = 'S'
    if direction == 'S':
        new_direction = 'N'
    if direction == 'W':
        new_direction = 'E'
    if direction == 'E':
        new_direction = 'W'
    return new_direction

def make_action(treasure_map, current_point, end_point, action, hist_path):
    # porusamy agenta
    next_point = move_agent(action=action, current_point=current_point)
    # sprawdzamy czy ten ruch jest dozwolony - czy nadal jest na mapie
    if is_no_the_map(next_point=next_point, treasure_map=treasure_map): # jesli ruch jest dozwolony
        changed_direction = False
        new_direction = action
        reward = reward_value(next_point=next_point, hist_path=hist_path, treasure_map=treasure_map) # nagroda za ten konkretny ruch
    else:
        # poruszamy sie w przeciwnym kierunku
        changed_direction = True
        new_direction = reverse_direction(direction=action)
        next_point = move_agent(action=new_direction, current_point=current_point)
        reward = reward_value(next_point=next_point, hist_path=hist_path, treasure_map=treasure_map) # nagroda za ten konkretny ruch
    
    if not (tuple(next_point) in hist_path):
        hist_path.append(tuple(current_point))
    next_state = get_current_state(current_point=next_point, treasure_map=treasure_map)
    return hist_path, next_point, reward, next_state, changed_direction, new_direction

def translate_num_direction(num_direction):
    'Funkcja pozwala przetlumaczyc numer kolumny na kierunek w ktorym ma sie poruszac agent'
    if num_direction == 0:
        return 'N'
    if num_direction == 1:
        return 'S'
    if num_direction == 2:
        return 'W'
    if num_direction == 3:
        return 'E'
    
def translate_string_direction(string_direction):
    'Funkcja pozwala przetlumaczyc kierunek swiata w ktorym ma sie poruszac agent na numer kolumny w tabeli nagrod'
    if string_direction == 'N':
        return 0
    if string_direction == 'S':
        return 1
    if string_direction == 'W':
        return 2
    if string_direction == 'E':
        return 3

def argmax_random(x):
    'Jesli max nie jest unikalny to wybierz losowy.'
    return np.random.choice(np.flatnonzero(x == x.max()))

def argmin_random(x):
    'Jesli min nie jest unikalny to wybierz losowy.'
    return np.random.choice(np.flatnonzero(x == x.min()))

def unpacking_Q_elem(tup):
    return tup[0], tup[1]

def q_learning(treasure_map, Q_elem, current_state, current_point, end_point, alpha = 0.1, gamma=0.9, epsilon=0.5):
    # Inicjujemy sumę nagrod:
    sum_of_rewards = 0
    
    # Inicjujemy sciezke (miejsca w których już byl)
    hist_path = []
    # Unpacking Q_elem
    Q_table, state_space_dict = unpacking_Q_elem(Q_elem)
    
    ###
#     print(treasure_map)
#     print('current_state na poczatku:',current_state)
#     print('current_point na poczatku:',current_point)
    ###
    
    treasure_found = False
    while not treasure_found:
        if random.uniform(0,1) <= epsilon: # jesli prawdopodob ruchu losowego > epsilon
            direction = random.choice(['N','S','W','E'])
        else:
            # jesli ruch nie losowy wybierz stan najbardziej korzystny
            # sprawdz czy dany stan już istnieje w słowniku
            if not (current_state in state_space_dict.values()): # jesli nie ma tego stanu w słowniku
                # dodaj do słownika
                state_space_dict[max(list(state_space_dict.keys()))+1] = current_state
                Q_table = Q_table = np.append(Q_table, [[0,0,0,0]], axis=0)

            # najpierw odkoduj ze slownika jaki wiersz w tabeli nagrod odpowiada aktualnemu stanowi
            state_key = list(state_space_dict.values()).index(current_state)
            # znajdz kierunek ktory jest najbardziej oplacalny
            direction = translate_num_direction(num_direction = argmax_random(Q_table[state_key]) )
        
        # Skoro juz mamy kierunek ustalony to przesuwamy agenta - dostajemy wtedy kolejny punkt, nagrode za ta akcje oraz nastepny stan
        hist_path, next_point, reward_for_action, next_state, changed_direction, new_direction = make_action(treasure_map=treasure_map, current_point=current_point, end_point=end_point, action=direction, hist_path=hist_path)
        sum_of_rewards = sum_of_rewards + reward_for_action
        if changed_direction:
            direction = new_direction
        if not (next_state in state_space_dict.values()): # jesli nie ma przyszłego stanu w słowniku
            # dodaj do słownika
            state_space_dict[max(list(state_space_dict.keys()))+1] = next_state
            Q_table = np.append(Q_table, [[0,0,0,0]], axis=0)
        
        # Tworzymy pomocnicze listy pomagajace odkodowac pozycje danego stanu w macierzy
        list_of_states_key = list(state_space_dict.keys())
        list_of_states_value = list(state_space_dict.values())
        
        ##
#         print('obecny punkt',current_point)
#         print('nowy kierunek:',direction)
#         print('kolejny punkt:',next_point)
#         print('kolejny stan:',next_state)
#         print('nagroda:',reward_for_action)
#         print('czy kierunek zmieniony:',changed_direction)
#         print('zmieniono kierunek na:',new_direction)
#         print('historia ruchow',hist_path)
#         print('\n')
        ##
        
        # sprawdzamy czy po wykonaniu ruchu skarb zostal znaleziony
        if is_treasure_found(current_point=next_point, end_point=end_point):
            treasure_found = True
            
            # Teraz aktualizacja tablicy nagrod
            q_value_current = Q_table[list_of_states_value.index(current_state)][translate_string_direction(string_direction=direction)]
            q_value_next = np.amax(Q_table[list_of_states_value.index(next_state)], axis=0)

            # uaktualniamy tabele nagrod
            q_value_new = (1-alpha)*q_value_current+alpha*(reward_for_action+gamma*q_value_next)
            Q_table[list_of_states_key[list_of_states_value.index(current_state)]][translate_string_direction(string_direction=direction)] = q_value_new
            break
        
        # Teraz aktualizacja tablicy nagrod
        q_value_current = Q_table[list_of_states_value.index(current_state)][translate_string_direction(string_direction=direction)]
        q_value_next = np.amax(Q_table[list_of_states_value.index(next_state)], axis=0)
        
        # uaktualniamy tabele nagrod
        q_value_new = (1-alpha)*q_value_current+alpha*(reward_for_action+gamma*q_value_next)
        Q_table[list_of_states_key[list_of_states_value.index(current_state)]][translate_string_direction(string_direction=direction)] = q_value_new
        
        # uaktualniamy punkty i stany
        current_point = next_point
        current_state = next_state
        
    # Packing
    Q_elem_return = (Q_table, state_space_dict)
    return Q_elem_return, sum_of_rewards

# Czego się nauczył agent?
def show_results(state_number, Q_elem):
    state = Q_elem[1][state_number]
    print('STATE:',state)
    print(np.concatenate((np.array(state)[:4], np.array(['X']),np.array(state)[4:] ), axis=0).reshape(3,3))
    print(Q_elem[0][state_number])
    print('\n')   

def play_game(treasure_map, Q_elem, start_point, end_point):
    # unpacking
    Q_table, state_space_dict = unpacking_Q_elem(Q_elem)
    
    # poczatkowa postac planszy
    gif_step = 0
    fig = plt.figure()
    plt.matshow(treasure_map)
    plt.scatter(start_point[1],start_point[0], color='r', s=250)
    plt.savefig(fname='gif/gif_step_'+str(gif_step)+'.png')
    plt.gca()
    plt.close(fig)
    time.sleep(1)
    gif_step = 1
    
    current_point = start_point
    
    while not all(current_point == end_point):
        if gif_step > 30:
            break
        
        current_state = get_current_state(current_point=current_point, treasure_map=treasure_map)
        # jesli nie ma danego stanu w slowniku musimy go dodac sztucznie
        if not (current_state in state_space_dict.values()): # jesli nie ma tego stanu w słowniku
                # dodaj do słownika
                state_space_dict[max(list(state_space_dict.keys()))+1] = current_state
                Q_table = Q_table = np.append(Q_table, [[0,0,0,0]], axis=0)
        # odkoduj ze slownika jaki wiersz w tabeli nagrod odpowiada aktualnemu stanowi
        state_key = list(state_space_dict.values()).index(current_state)
        # znajdz kierunek ktory jest najbardziej oplacalny
        direction = translate_num_direction(num_direction = argmax_random(Q_table[state_key]) )   
        # porusz agenta w tym kierunku
        next_point = move_agent(action=direction, current_point=current_point)
        
        if not is_no_the_map(next_point=next_point, treasure_map=treasure_map):
            while not is_no_the_map(next_point=next_point, treasure_map=treasure_map): # dopoki kolejny punkt jest poza mapą
                # wybierz drugi, trzeci itp najlepszy kierunek
                temp_Q_table = Q_table[state_key]
                temp_Q_table[translate_string_direction(direction)] = temp_Q_table.min() # najlepszy (niemożliwy ruch) tymczasowo ustawiamy na najmniejszy aby moc wybrac inny najlepszy
                # znajdz inny kierunek ktory jest najbardziej oplacalny
                direction = translate_num_direction(num_direction = argmax_random(temp_Q_table))
                # porusz agenta w tym kierunku
                next_point = move_agent(action=direction, current_point=current_point)
        
        # uaktualnij rysunek
        fig = plt.figure()
        plt.matshow(treasure_map)
        plt.scatter(next_point[1],next_point[0], color='r', s=250)
        plt.savefig(fname='gif/gif_step_'+str(gif_step)+'.png')
        plt.gca()
        plt.close(fig)
        time.sleep(1)
        
        current_point = next_point
        gif_step += 1

