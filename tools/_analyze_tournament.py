import csv, collections, numpy as np

with open('runs/models_7x7/checkpoints/tournament_results_matchups.csv', newline='') as f:
    rows = list(csv.DictReader(f))

elo_order = [
    'runs/models_7x7\\supervised_extended.pt',
    'cycle_0111','cycle_0091','cycle_0101','cycle_0071',
    'cycle_0081','cycle_0121','cycle_0141','cycle_0131',
    'cycle_0061','cycle_0051','cycle_0041',
    'runs/models_7x7_v2\\best.pt',
    'cycle_0031','minimax-3','minimax-2',
    'cycle_0021','cycle_0011','cycle_0001',
]

pair_scores = collections.defaultdict(list)
for row in rows:
    pair_scores[(row['agent_a'], row['agent_b'])].append(float(row['score_for_a']))

def score_pct(a, b):
    s1 = pair_scores.get((a, b), [])
    s2 = [1 - x for x in pair_scores.get((b, a), [])]
    all_s = s1 + s2
    return (sum(all_s) / len(all_s) * 100) if all_s else None

lbl = {}
for n in elo_order:
    s = n.replace('runs/models_7x7\\', '').replace('runs/models_7x7_v2\\', 'v2/').replace('.pt', '')
    lbl[n] = s

# ── 1. Head-to-head matrix (top 9) ───────────────────────────────────────────
top9 = elo_order[:9]
print('HEAD-TO-HEAD SCORE% (row beats col)\n')
print(f"{'':>20}" + ''.join(f'{lbl[c]:>10}' for c in top9))
for a in top9:
    row_str = f'{lbl[a]:>20}'
    for b in top9:
        if a == b:
            row_str += f'{"---":>10}'
        else:
            s = score_pct(a, b)
            row_str += (f'{s:>9.1f}%' if s is not None else f'{"N/A":>10}')
    print(row_str)

# ── 2. Tier analysis ──────────────────────────────────────────────────────────
weak   = elo_order[13:]   # cycle_0031 and below (inc minimax)
mid    = elo_order[6:13]  # cycle_0121 through v2/best
strong = elo_order[:6]    # supervised_extended + top 5 RL models

def avg_vs(a, peers):
    vals = [score_pct(a, b) for b in peers if b != a]
    vals = [v for v in vals if v is not None]
    return np.mean(vals) if vals else float('nan')

print('\n\nSCORE% BY OPPONENT TIER')
print(f"  Tiers: strong=rank1-6, mid=rank7-13, weak=rank14-19")
print(f"\n{'Agent':>22}  {'vs strong':>10}  {'vs mid':>8}  {'vs weak':>8}")
for a in elo_order[:9]:
    print(f'{lbl[a]:>22}  {avg_vs(a, strong):>9.1f}%  {avg_vs(a, mid):>7.1f}%  {avg_vs(a, weak):>7.1f}%')

# ── 3. First-player advantage per agent ──────────────────────────────────────
print('\n\nP1 WIN RATE (as player 1) vs P2 WIN RATE (as player 2)')
print(f"{'Agent':>22}  {'as P1':>8}  {'as P2':>8}  {'total':>8}")
for a in elo_order[:9]:
    p1_games = [row for row in rows if row['player1'] == a]
    p2_games = [row for row in rows if row['player2'] == a]
    def win_rate(game_list, player):
        if not game_list: return float('nan')
        wins = sum(1 for r in game_list if r['result'] == ('p1_win' if player == 'player1' else 'p2_win'))
        draws = sum(1 for r in game_list if r['result'] == 'draw')
        return (wins + 0.5 * draws) / len(game_list) * 100
    total = p1_games + p2_games
    total_rate = (sum(1 for r in total if (r['result']=='p1_win' and r['player1']==a) or
                                           (r['result']=='p2_win' and r['player2']==a))
                  + 0.5 * sum(1 for r in total if r['result']=='draw')) / len(total) * 100 if total else float('nan')
    print(f'{lbl[a]:>22}  {win_rate(p1_games,"player1"):>7.1f}%  {win_rate(p2_games,"player2"):>7.1f}%  {total_rate:>7.1f}%')

# ── 4. Late cycles vs specifically the ones they should beat ─────────────────
print('\n\nLATE CYCLES (0111-0141) vs WEAK/MID IN DETAIL')
late = ['cycle_0111','cycle_0121','cycle_0131','cycle_0141']
victims = ['cycle_0031','minimax-3','minimax-2','cycle_0021','cycle_0011','cycle_0001',
           'cycle_0061','cycle_0051','cycle_0041']
print(f"{'':>12}" + ''.join(f'{lbl[b]:>10}' for b in victims))
for a in late:
    row_str = f'{lbl[a]:>12}'
    for b in victims:
        s = score_pct(a, b)
        row_str += (f'{s:>9.1f}%' if s is not None else f'{"N/A":>10}')
    print(row_str)
