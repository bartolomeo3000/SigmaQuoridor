import copy
import time
from tqdm import tqdm

import traceback
from pathlib import Path
import joblib

import multiprocessing
from multiprocessing import Pool, cpu_count #, Manager
from multiprocessing.pool import ThreadPool


import numpy as np
import pandas as pd

from game import State
from rl_models import BOARDSIZE, QLearningAgent, SarsaAgent, ExpectedSarsaAgent, DoubleQLearningAgent, DoubleSarsaAgent, DoubleExpectedSarsaAgent

from mcts import MCTSAgent
from dual_network import NNEvaluator, load_model
from benchmark_agents import RandomAgent, GreedyDistanceAgent


#BOARDSIZE = 5
WALLS = 1

SEED = 0
PROCESSES = cpu_count() - 1

DRAW_REWARD = -0.2 # penalty for being risk-averse
WIN_REWARD = 1.0
LOSS_REWARD = -1.0

MODELS_DIR = Path(f"models_{BOARDSIZE}x{BOARDSIZE}_with_{WALLS}_walls")
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
BEST_DIR = MODELS_DIR / "best"

REFERENCE_MODEL_PATH = Path("models_5x5/supervised_extended.pt")
REFERENCE_MODEL_SCORES_PATH = BEST_DIR / "scores"
REFERENCE_MODEL_NUM_SIMULATIONS = 1

GLOBAL_EVAL_AGENT_A = None
GLOBAL_EVAL_AGENT_B = None
GLOBAL_RESULTS = None

def make_run_dir(agent_a, agent_b, run_name=None):
    if run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{agent_a.__class__.__name__}_vs_{agent_b.__class__.__name__}__{timestamp}"

    return CHECKPOINT_DIR / run_name

def find_latest_epoch(run_dir):
    if not run_dir.exists():
        return None

    epoch_dirs = sorted(run_dir.glob("epoch_*"))
    if not epoch_dirs:
        return None

    return int(epoch_dirs[-1].name.split("_")[1])

def save_checkpoint(run_dir, worker_id, agent_a, agent_b, epoch):
    worker_dir = run_dir / f"worker_{worker_id}"
    epoch_dir = worker_dir / f"epoch_{epoch:06d}"
    epoch_dir.mkdir(parents=True, exist_ok=True,)
    agent_a.save(epoch_dir / "agent_a.pkl")
    agent_b.save(epoch_dir / "agent_b.pkl")

def load_checkpoint(run_dir, worker_id, agent_a, agent_b):
    worker_dir = run_dir / f"worker_{worker_id}"
    latest_epoch = find_latest_epoch(worker_dir)

    if latest_epoch is None:
        return agent_a, agent_b, 0

    print(f"[Worker {worker_id}] Resuming from epoch {latest_epoch}")

    epoch_dir = worker_dir / f"epoch_{latest_epoch:06d}"
    agent_a.load(epoch_dir / "agent_a.pkl")
    agent_b.load(epoch_dir / "agent_b.pkl")

    return agent_a, agent_b, latest_epoch

def save_compressed_to_uncompressed(compressed_path, uncompressed_path=None):
    compressed_path = Path(compressed_path)
    if uncompressed_path is None:
        if compressed_path.stem.endswith("_compressed"):
            uncompressed_path = (compressed_path.with_name(compressed_path.stem[:-11] + compressed_path.suffix) if compressed_path.stem.endswith("_compressed") else
                                 compressed_path.with_name(compressed_path.stem + "_uncompressed" + compressed_path.suffix))

    print('Loading compressed agent')
    data = joblib.load(compressed_path)
    print('Saving uncompressed agent')
    joblib.dump(data, uncompressed_path)
    return Path(uncompressed_path)

def save_uncompressed_to_compressed(uncompressed_path, compressed_path=None):
    uncompressed_path = Path(uncompressed_path)
    if compressed_path is None:
        compressed_path = uncompressed_path.with_name(uncompressed_path.stem + "_compressed" + uncompressed_path.suffix)

    print('Loading uncompressed agent')
    data = joblib.load(uncompressed_path)
    print('Saving compressed agent')
    joblib.dump(data, compressed_path, compress=('gzip', 3))
    return Path(compressed_path)


def load_best_models(agent_a, agent_b):
    agent_a_name = agent_a.__name__
    agent_b_name = agent_b.__name__ if agent_b is not None else None

    agent_a_path = BEST_DIR / f"{agent_a_name}.pkl"
    agent_b_path = BEST_DIR / f"{agent_b_name}.pkl" if agent_b is not None else REFERENCE_MODEL_PATH

    agent_a_compressed_path = BEST_DIR / f"{agent_a_name}_compressed.pkl"
    agent_b_compressed_path = BEST_DIR / f"{agent_b_name}_compressed.pkl" if agent_b is not None else None


    a_exists = agent_a_path.exists() or agent_a_compressed_path.exists()
    if not a_exists:
        print(f"No saved model for {agent_a_name}, starting fresh.")

    if agent_b is not None:
        b_exists = agent_b_path.exists() or agent_b_compressed_path.exists()
        if not b_exists:
            print(f"No saved model for {agent_b_name}, starting fresh.")

    agent_a = agent_a()
    if a_exists:
        if not agent_a_path.exists():
            print(f"Missing uncompressed model for {agent_a_name}, falling back to compressed version.")
        agent_a.load(agent_a_path if agent_a_path.exists() else agent_a_compressed_path)

    if agent_b is not None:
        agent_b = agent_b()
        if b_exists:
            if not agent_b_path.exists():
                print(f"Missing uncompressed model for {agent_b_name}, falling back to compressed version.")
            agent_b.load(agent_b_path if agent_b_path.exists() else agent_b_compressed_path)
    else:
        model = load_model(agent_b_path)
        model.eval()
        agent_b = MCTSAgent(evaluator=NNEvaluator(model), num_simulations=REFERENCE_MODEL_NUM_SIMULATIONS, training=False)

    return agent_a, agent_b
    

# def init_worker(agent1, agent2):
#     global GLOBAL_AGENT1
#     global GLOBAL_AGENT2
#
#     GLOBAL_AGENT1 = copy.deepcopy(agent1)
#     GLOBAL_AGENT2 = copy.deepcopy(agent2)


def self_play_episode(agent_a, agent_b, boardsize, walls):
    state = State(boardsize=boardsize, walls_p1=walls, walls_p2=walls)

    swap = agent_a.rng.random() < 0.5
    agent1 = agent_b if swap else agent_a
    agent2 = agent_a if swap else agent_b

    while not state.is_finished():
        current_player = state.get_current_player()
        agent = agent1 if current_player == 1 else agent2

        action = agent.select_action(state)
        next_state = state.next(action)

        is_terminal_state = next_state.is_finished()
        if is_terminal_state:
            winner = next_state.winner()
            reward = DRAW_REWARD if winner == 0 else (WIN_REWARD if winner == current_player else LOSS_REWARD)
        else:
            reward = 0.0

        agent.update(state, action, reward, next_state, is_terminal_state)
        state = next_state

def play(agent_a, agent_b, boardsize, walls):
    state = State(boardsize=boardsize, walls_p1=walls, walls_p2=walls)
    while not state.is_finished():
        agent = agent_a if state.get_current_player() == 1 else agent_b
        state = state.next(agent.select_action(state, training=False))

    return state.winner() #"depth": state.depth} # depth is used to determine how long the game has progressed and which player's turn it is

def init_eval_worker(agent_a, agent_b):
    global GLOBAL_EVAL_AGENT_A
    global GLOBAL_EVAL_AGENT_B

    GLOBAL_EVAL_AGENT_A = agent_a
    GLOBAL_EVAL_AGENT_B = agent_b

def evaluate_one(args):
    i, boardsize, walls = args

    global GLOBAL_EVAL_AGENT_A
    global GLOBAL_EVAL_AGENT_B

    try:
        if i % 2 == 0:
            winner = play(GLOBAL_EVAL_AGENT_A, GLOBAL_EVAL_AGENT_B, boardsize, walls)
            if winner == 1:
                return "A"
            elif winner == 2:
                return "B"
            return "draw"

        winner = play(GLOBAL_EVAL_AGENT_B, GLOBAL_EVAL_AGENT_A, boardsize, walls)
        if winner == 1:
            return "B"
        elif winner == 2:
            return "A"
        return "draw"
    
    except Exception as _:
        print(f"Error: worker={i}\n{traceback.format_exc()}")

def evaluate(agent_a, agent_b, boardsize, walls, games, eval_mode=False, processes=PROCESSES):
    wins = {"A": 0, "B": 0, "draw": 0}

    if not eval_mode:
        for i in range(games):
            if i % 2 == 0:
                winner = play(agent_a, agent_b, boardsize, walls)
                if winner == 1:
                    wins["A"] += 1
                elif winner == 2:
                    wins["B"] += 1
                else:
                    wins["draw"] += 1
            else:
                winner = play(agent_b, agent_a, boardsize, walls)
                if winner == 1:
                    wins["B"] += 1
                elif winner == 2:
                    wins["A"] += 1
                else:
                    wins["draw"] += 1

        return wins

    multiprocessing.freeze_support()
    with Pool(processes=processes, initializer=init_eval_worker, initargs=(agent_a, agent_b)) as pool:
        results = list(tqdm(pool.imap_unordered(evaluate_one, [(i, boardsize, walls) for i in range(games)]), total=games, dynamic_ncols=True))

    filtered = [r for r in results if r is not None]
    print(f"Completed: {len(filtered)}/{len(results)} tasks in evaluation")

    for result in filtered:
        wins[result] += 1

    return wins

def maybe_update_best(current_agent_a, current_agent_b, best_agent_a, best_agent_b, boardsize, walls, games):
    r1 = evaluate(current_agent_a, best_agent_a, boardsize, walls, games=games)
    r2 = evaluate(current_agent_b, best_agent_b, boardsize, walls, games=games)

    winrate1 = (r1["A"] + 0.5 * r1["draw"]) / (r1["A"] + r1["B"] + r1["draw"])
    winrate2 = (r2["A"] + 0.5 * r2["draw"]) / (r2["A"] + r2["B"] + r2["draw"])

    if winrate1 > 0.5:
        best_agent_a = copy.deepcopy(current_agent_a)
    if winrate2 > 0.5:
        best_agent_b = copy.deepcopy(current_agent_b)

    return best_agent_a, best_agent_b, winrate1, winrate2

def init_pool_worker(results=None):
    global GLOBAL_RESULTS
    GLOBAL_RESULTS = results

def train_worker(args):
    worker_id, run_dir, agent_a, agent_b, epochs, boardsize, walls, checkpoint_every = args

    rng = np.random.default_rng(SEED + worker_id)
    max_seed = 2**31 - 1
    seed1 = int(rng.integers(0, max_seed))
    seed2 = int(rng.integers(0, max_seed))

    agent_a.set_seed(seed1)
    agent_b.set_seed(seed2)

    try:
        agent_a, agent_b, start_epoch = load_checkpoint(run_dir, worker_id, agent_a, agent_b)

        iterator = tqdm(range(start_epoch, epochs), leave=False) if worker_id == 0 else range(start_epoch, epochs)
        for epoch in iterator:
            self_play_episode(agent_a, agent_b, boardsize, walls)

            if (epoch + 1) % checkpoint_every == 0:
                save_checkpoint(run_dir, worker_id, agent_a, agent_b, epoch + 1)
                print(f"\nCheckpoint saved at epoch {epoch + 1}/{epochs}")

        return {"worker_id": worker_id, "agent_a": agent_a, "agent_b": agent_b, 'score1': None, 'score2': None}
    
    except Exception as _:
        print(f"Error: worker={worker_id}, agent_a={agent_a}, agent_b={agent_b}\n{traceback.format_exc()}")


def tournament_match_worker(args):
    i, j, boardsize, walls, games = args

    global GLOBAL_RESULTS
    ai_a = GLOBAL_RESULTS[i]["agent_a"]
    aj_a = GLOBAL_RESULTS[j]["agent_a"]
    ai_b = GLOBAL_RESULTS[i]["agent_b"]
    aj_b = GLOBAL_RESULTS[j]["agent_b"]

    try:
        r1 = evaluate(ai_a, aj_a, boardsize, walls, games)
        score1 = (r1["A"] + 0.5 * r1["draw"]) / (r1["A"] + r1["B"] + r1["draw"])

        r2 = evaluate(ai_b, aj_b, boardsize, walls, games)
        score2 = (r2["A"] + 0.5 * r2["draw"]) / (r2["A"] + r2["B"] + r2["draw"])

        #return i, score1 + score2, 2
        return i, j, score1 + score2, (1.0 - score1) + (1.0 - score2), 2

    except Exception as _:
        print(f"Error: tournament i={i}, j={j}\n{traceback.format_exc()}")

def tournament(results, boardsize, walls, games, pool):
    n = len(results)
    scores = [0.0 for _ in range(n)]
    matches = [0 for _ in range(n)]

    worker_args = [(i, j, boardsize, walls, games) for i in range(n) for j in range(i + 1, n)]
    tournament_results = list(tqdm(pool.imap_unordered(tournament_match_worker, worker_args), leave=False, total=len(worker_args), dynamic_ncols=True))

    filtered = [r for r in tournament_results if r is not None]
    print(f"Completed: {len(filtered)}/{len(tournament_results)} tasks in tournament")

    for result in filtered:
        i, j, score_i, score_j, match_count = result
        scores[i] += score_i
        scores[j] += score_j
        matches[i] += match_count
        matches[j] += match_count

    final_scores = [scores[i] / matches[i] if matches[i] > 0 else 0.0 for i in range(n)]
    return results[max(range(len(final_scores)), key=lambda k: final_scores[k])], final_scores

def update_global_best(new_agent_a, new_agent_b, boardsize, walls, games,
                       benchmark_sigma=True, benchmark_random=True, benchmark_greedy=True,
                       games_sigma=None, games_random=None, games_greedy=None):
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    new_agent_a_name = new_agent_a.__class__.__name__
    new_agent_b_name = new_agent_b.__class__.__name__ if new_agent_b is not None else None

    agent_a_path = BEST_DIR / f"{new_agent_a_name}.pkl"
    agent_b_path = BEST_DIR / f"{new_agent_b_name}.pkl" if new_agent_b is not None else REFERENCE_MODEL_PATH

    agent_a_compressed_path = BEST_DIR / f"{new_agent_a_name}_compressed.pkl"
    agent_b_compressed_path = BEST_DIR / f"{new_agent_b_name}_compressed.pkl" if new_agent_b is not None else None

    replace_agent_a = True
    existing_agent_a_path = agent_a_path if agent_a_path.exists() else agent_a_compressed_path

    if existing_agent_a_path.exists():
        old_agent_a = copy.deepcopy(new_agent_a)

        if existing_agent_a_path == agent_a_compressed_path:
            print(f"Missing uncompressed model for {new_agent_a_name}, falling back to compressed version.")

        old_agent_a.load(existing_agent_a_path)

        result = evaluate(new_agent_a, old_agent_a, boardsize, walls, games, eval_mode=True)
        winrate = (result["A"] + 0.5 * result["draw"]) / (result["A"] + result["B"] + result["draw"])

        print(f"\n[AgentA] New vs Old winrate: {winrate:.4f}")
        replace_agent_a = winrate > 0.5

    if new_agent_b is None:
        score = None
        score_vs_random = None
        score_vs_greedy = None

        if benchmark_sigma:
            model = load_model(agent_b_path)
            model.eval()
            reference_agent = MCTSAgent(evaluator=NNEvaluator(model), num_simulations=REFERENCE_MODEL_NUM_SIMULATIONS, training=False)
            result = evaluate(new_agent_a, reference_agent, boardsize, walls, games_sigma if games_sigma is not None else games, eval_mode=True)
            score = (result["A"] + 0.5 * result["draw"]) / (result["A"] + result["B"] + result["draw"])

        if benchmark_random:
            result_random = evaluate(new_agent_a, RandomAgent(), boardsize, walls, games_random if games_random is not None else games)
            score_vs_random = (result_random["A"] + 0.5 * result_random["draw"]) / (result_random["A"] + result_random["B"] + result_random["draw"])

        if benchmark_greedy:
            result_greedy = evaluate(new_agent_a, GreedyDistanceAgent(), boardsize, walls, games_greedy if games_greedy is not None else games)
            score_vs_greedy = (result_greedy["A"] + 0.5 * result_greedy["draw"]) / (result_greedy["A"] + result_greedy["B"] + result_greedy["draw"])

        if Path(REFERENCE_MODEL_SCORES_PATH).exists():
            df = pd.read_csv(REFERENCE_MODEL_SCORES_PATH)
        else:
            df = pd.DataFrame(columns=["model", "score_vs_sigma", "score_vs_random", "score_vs_greedy"])
        row = df[df["model"] == new_agent_a_name]
        current_best_score = -np.inf if (row.empty or "score_vs_greedy" not in row.columns or pd.isna(row.iloc[0]["score_vs_greedy"])) else np.float64(row.iloc[0]["score_vs_greedy"])

        if score is not None:
            print(f"\n[Sigma ref]   {new_agent_a_name} vs reference agent:     {score:.4f}")
        if score_vs_random is not None:
            print(f"[Random]      {new_agent_a_name} vs random agent:          {score_vs_random:.4f}")
        if score_vs_greedy is not None:
            print(f"[Greedy dist] {new_agent_a_name} vs greedy-distance agent: {score_vs_greedy:.4f}")

        # Gate on greedy score; if greedy was not run, always save
        gate_score = score_vs_greedy if score_vs_greedy is not None else np.inf

        if gate_score >= current_best_score:
            new_agent_a.save(agent_a_path)

            new_row = {"model": new_agent_a_name}
            if score is not None:           new_row["score_vs_sigma"]  = score
            if score_vs_random is not None: new_row["score_vs_random"] = score_vs_random
            if score_vs_greedy is not None: new_row["score_vs_greedy"] = score_vs_greedy

            if row.empty:
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                for col, val in new_row.items():
                    if col != "model":
                        df.loc[df["model"] == new_agent_a_name, col] = val
            df.to_csv(REFERENCE_MODEL_SCORES_PATH, index=False)

            print(f"[Greedy dist] Updated best model: {agent_a_path}")
        else:
            print("[Greedy dist] Keeping previous best.")

        return

    if replace_agent_a:
        new_agent_a.save(agent_a_path)
        print(f"[AgentA] Updated global best: {agent_a_path}")
    else:
        print("[AgentA] Keeping previous best.")

    replace_agent_b = True
    existing_agent_b_path = agent_b_path if agent_b_path.exists() else agent_b_compressed_path

    if existing_agent_b_path.exists():
        old_agent_b = copy.deepcopy(new_agent_b)

        if existing_agent_b_path == agent_b_compressed_path:
            print(f"Missing uncompressed model for {new_agent_b_name}, falling back to compressed version.")

        old_agent_b.load(existing_agent_b_path)

        result = evaluate(new_agent_b, old_agent_b, boardsize, walls, games, eval_mode=True)
        winrate = (result["A"] + 0.5 * result["draw"]) / (result["A"] + result["B"] + result["draw"])

        print(f"\n[AgentB] New vs Old winrate: {winrate:.4f}")
        replace_agent_b = winrate > 0.5

    if replace_agent_b:
        new_agent_b.save(agent_b_path)
        print(f"[AgentB] Updated global best: {agent_b_path}")
    else:
        print("[AgentB] Keeping previous best.")


def train_population(agent_a=None, agent_b=None, epochs=5000, boardsize=BOARDSIZE, walls=WALLS, processes=PROCESSES, checkpoint_every=20000, tournament_games=100,
                     run_name=None, compare_with_reference_model=False,
                     benchmark_sigma=True, benchmark_random=True, benchmark_greedy=True,
                     games_sigma=None, games_random=None, games_greedy=None):
    if agent_a is None:
        agent_a = QLearningAgent()
    if agent_b is None:
        agent_b = QLearningAgent()

    multiprocessing.freeze_support()
    run_dir = make_run_dir(agent_a, agent_b, run_name=run_name)
    print(f"Run dir: {run_dir}")

    worker_args = [(worker_id, run_dir, copy.deepcopy(agent_a), copy.deepcopy(agent_b), epochs, boardsize, walls, checkpoint_every)
                   for worker_id in range(processes)]

    # shared_results = Manager().list([None for _ in range(processes)])
    # with Pool(processes=processes, initializer=init_pool_worker, initargs=(agent1, agent2, shared_results)) as pool:
    #     results_raw = list(tqdm(pool.imap_unordered(train_worker, worker_args), total=len(worker_args), dynamic_ncols=True))

    #     results = [r for r in list(shared_results) if r is not None]
    #     print(f"Completed: {len(results)}/{len(results_raw)} tasks in training")

    #     print("\nRunning tournament...")
    #     best_result, scores = tournament(results, boardsize, walls, tournament_games, pool=pool)

    with Pool(processes=processes) as pool:
        results_raw = list(tqdm(pool.imap_unordered(train_worker, worker_args), total=len(worker_args), dynamic_ncols=True))

    results = [r for r in results_raw if r is not None]
    print(f"Completed: {len(results)}/{len(results_raw)} tasks in training")

    if not results:
        print("All workers failed — skipping tournament and model update.")
        return agent_a, agent_b

    print("\nRunning tournament...")
    with ThreadPool(processes=processes, initializer=init_pool_worker, initargs=(results,)) as pool:
        best_result, scores = tournament(results, boardsize, walls, tournament_games, pool=pool)

    print("\nTournament results:")
    for result, score in zip(results, scores):
        print(f"Worker {result['worker_id']}: score={score:.4f}")
        r1 = evaluate(result["agent_a"], best_result["agent_a"], boardsize, walls, tournament_games)
        r2 = evaluate(result["agent_b"], best_result["agent_b"], boardsize, walls, tournament_games)
        print(f"\tagent_a: W={r1['A']} | L={r1['B']} | D={r1['draw']} | score={(r1['A'] + 0.5 * r1['draw']) / (r1['A'] + r1['B'] + r1['draw']):.4f}")
        print(f"\tagent_b: W={r2['A']} | L={r2['B']} | D={r2['draw']} | score={(r2['A'] + 0.5 * r2['draw']) / (r2['A'] + r2['B'] + r2['draw']):.4f}")

    print(f"\nBEST WORKER: {best_result['worker_id']}")
    update_global_best(best_result["agent_a"], None if compare_with_reference_model else best_result["agent_b"], boardsize, walls, games=10,
                       benchmark_sigma=benchmark_sigma, benchmark_random=benchmark_random, benchmark_greedy=benchmark_greedy,
                       games_sigma=games_sigma, games_random=games_random, games_greedy=games_greedy)

    return best_result["agent_a"], best_result["agent_b"]



if __name__ == "__main__":
    NUM_CYCLES = 20
    EPOCHS_PER_CYCLE = 5000

    a1, a2 = load_best_models(DoubleQLearningAgent, DoubleSarsaAgent)

    for cycle in range(1, NUM_CYCLES + 1):
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle}/{NUM_CYCLES}")
        print(f"{'='*60}")
        a1, a2 = train_population(a1, a2, epochs=EPOCHS_PER_CYCLE, compare_with_reference_model=True, benchmark_sigma=False,
                                   games_random=400, games_greedy=10)