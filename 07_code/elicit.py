
import json, re, numpy as np, commons

SYSTEM = ("You are a participant in an online economic decision-making study. Make your own decisions as a "
          "participant in this study. Reply only with the JSON object requested, with no other text.")

INSTR = {}
INSTR['flowers'] = """INVESTMENT GAME - INSTRUCTIONS
You are one of 4 players in an investment game. You can earn points depending on your own decisions and the decisions of the others. You receive a base payment for completing the task and a bonus that depends on the number of points you earn.

There is a field of flowers (the pool) that is controlled by a manager. The field starts with 200 flowers and can never hold more than 200 flowers.
Each round, the manager offers flowers from the field to each of the 4 players. The manager can offer different amounts to different players, and everyone can see what everyone was offered.
You then decide how much of your offer to re-invest in the field (one flower = one coin; you choose whole coins). The coins that you re-invest grow by 40% (they are multiplied by 1.4) and are added back to the field, but the field is capped at 200 flowers. Whatever you do not re-invest you keep as points.
The other three players make the same decision at the same time. If the field falls to zero it cannot recover and nobody receives anything any more.
At the end of each round you see how the field changed, what every player was offered and how much every player re-invested.

You do not know how many rounds the game lasts. You will be paid in proportion to the points you keep between round 1 and a randomly chosen round of the game. This means you should continuously play as if the game could end at any time. Because of this, be aware that the further you are into the game, the less likely you are to actually be paid the money earned late in the game."""

INSTR['neutral'] = """DECISION TASK - INSTRUCTIONS
You are one of four members of a group. Your bonus payment depends on the points you accumulate, which depend on your own choices and on the choices of the other members.

The group shares a common fund that is administered by an allocator. The fund begins at 200 units and is capped at 200 units.
In every period the allocator distributes units from the fund to the four members. Allocations may differ across members and all allocations are visible to everyone.
Each member then chooses how many of the units allocated to them to return to the fund (whole units only). Every returned unit is multiplied by 1.4 and added to the fund, up to the cap of 200. Units that are not returned become the member's points.
All members choose simultaneously. A fund that reaches zero stays at zero, and no further allocations are possible.
After every period all members see the new size of the fund, every member's allocation and every member's return.

The number of periods is not announced. Your bonus is proportional to the points you earn from period 1 up to a period drawn at random, so you should behave as if the task might stop at any moment; points earned in later periods are less likely to count."""

EXPL = {
 'prop': 'The manager will offer flowers proportional to the last re-investment. For example, if half of the total reinvestment last round was done by you, you get half of the flowers this round. If a player is the only one who re-invested, they will get all the flowers next round. Generally, the more you re-invest the more you get offered, but it is always relative to other players.',
 'interp': 'The manager will adjust its policy to the flower field size. When there are a lot of flowers in the field, the manager will tend to give flowers to everyone, no matter how much or little they reinvested. If there are few flowers in the field, the manager will tend to give more flowers to those players who re-invested the most on the last trial (relative to other players.)',
 'rl': 'The manager you will play with, has the following strategy: The manager aims to offer flowers in such a way that all players make the maximum amount of money possible over the course of the experiment.',
}
RULE = {
 'equal': 'Every round the manager offers the whole field, split equally among the four players, whatever anyone re-invested.',
 'mixed': 'Every round the manager offers the whole field. Half of it is split equally among the four players; the other half is split in proportion to what each player re-invested in the previous round.',
 'prop': 'Every round the manager offers the whole field, split in proportion to what each player re-invested in the previous round (a player who re-invested nothing gets nothing). In round 1 the field is split equally.',
 'interp': 'Every round the manager offers the whole field. A fraction w of it is split equally and the rest in proportion to what each player re-invested in the previous round, where w = (field size / 200) to the power 22, rounded to one decimal. So w is 1 with a full field (200), about 0.5 at 194, 0.1 at 175 and 0 below about 170; with a field below 170 the split is purely proportional to last round\'s re-investments.',
 'step': 'Every round the manager offers the whole field. If the field holds at least 194 flowers, it is split equally among the four players; otherwise it is split in proportion to what each player re-invested in the previous round.',
 'm2': EXPL['rl'],
}

def scenario_table():
    lines = ['ID | flowers in field now | offered to you | offered to the other three players (total) | other three re-invested last round (% of their offers) | you re-invested last round (% of your offer)']
    ids = []
    for k, (a, b, c, d, R, s, q, y) in enumerate(commons.cell_list()):
        mine = s * R / 4; others = R - mine
        sid = 'S%02d' % (k + 1); ids.append(sid)
        lines.append('%s | %d | %d | %d | %d%% | %d%%' % (sid, round(R), round(mine), round(others), round(q * 100), round(y * 100)))
    return '\n'.join(lines), ids

def build_prompt(cond, persona=None, mech=None):
    frame = 'neutral' if cond == 'neutral' else 'flowers'
    txt = INSTR[frame]
    if cond.startswith('expl_'):
        txt += "\n\nBefore the game you are told the manager's strategy: \"" + EXPL[cond[5:]] + '"'
    if cond == 'strategic':
        txt += ("\n\nYou are told exactly how the manager decides its offers: " + RULE[mech] +
                "\nYour only goal is to earn as many points for yourself as possible. The other three players are ordinary participants in the study.")
    if cond == 'persona' and persona:
        txt += '\n\nAnswer as the following participant would: ' + persona
    tab, ids = scenario_table()
    if frame == 'neutral':
        tab = tab.replace('flowers in field now', 'units in fund now').replace('offered to', 'allocated to').replace('re-invested', 'returned')
    ask = ("\n\nSTRATEGY QUESTIONNAIRE\nInstead of playing round by round, tell us in advance what you would do in each situation below. "
           "Any of these situations can occur in the game, so treat each one as a real decision. For each situation give the percentage "
           "(a whole number from 0 to 100) of your current offer that you would re-invest.\n\n"
           "F: round 1 of the game. The field holds 200 flowers and each of the four players is offered 50 flowers.\n\n"
           "Situations in later rounds:\n" + tab +
           "\n\nReply with one JSON object that has the key \"F\" and the keys S01 to S81, each mapped to an integer from 0 to 100, for example {\"F\": 0, \"S01\": 0, ...}.")
    if frame == 'neutral':
        ask = ask.replace('flowers', 'units').replace('field', 'fund').replace('offered', 'allocated').replace('re-invest', 'return').replace('offer', 'allocation')
    return txt + ask, ids

def parse(text, ids):
    m = re.search(r'\{.*\}', text, re.S)
    if not m: return None
    try: d = json.loads(m.group(0))
    except Exception: return None
    try:
        vals = [float(d[i]) for i in ids]; f = float(d['F'])
    except Exception: return None
    arr = np.clip(np.array(vals) / 100.0, 0, 1)
    return float(np.clip(f / 100.0, 0, 1)), arr

PERSONAS_AGE = [19, 23, 27, 31, 36, 42, 48, 55, 61, 67]
PERSONAS_JOB = ['student', 'nurse', 'software developer', 'retail assistant', 'teacher', 'accountant', 'electrician', 'unemployed job seeker', 'retired civil servant', 'graphic designer', 'warehouse worker', 'small-business owner']
PERSONAS_CTRY = ['the United Kingdom', 'the United States']
def persona(rng):
    return 'a %d-year-old %s from %s who takes part in online studies on Prolific.' % (rng.choice(PERSONAS_AGE), rng.choice(PERSONAS_JOB), rng.choice(PERSONAS_CTRY))
