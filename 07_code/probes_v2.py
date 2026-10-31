
"""probes_v2.py - direct-play probes with exact display precision (revision), matched-pair probes and a fully neutral strategy prompt."""
import numpy as np, re, json, elicit, commons
NAMES = ['You', 'Player 2', 'Player 3', 'Player 4']
JSON_ONLY = 'Give only the JSON object, without explanation.'
def f1(x): return ('%.1f' % x).rstrip('0').rstrip('.') if abs(x - round(x)) < 1e-9 else '%.1f' % x
def probe_text(t, pool_prev, offers_prev, reinv_prev, pool_now, offers_now):
    """offers/reinv lists ordered with the focal player first."""
    o = offers_now[0]; mx = int(np.floor(o + 1e-9))
    lines = ["GAME IN PROGRESS", "You are now playing round %d of the game (you do not know how many rounds are left)." % (t + 1),
             "Overview of the previous round: the field held %s flowers." % f1(pool_prev),
             "Offers: " + "; ".join("%s %s" % (n, f1(v)) for n, v in zip(NAMES, offers_prev)),
             "Re-invested: " + "; ".join("%s %d" % (n, int(round(v))) for n, v in zip(NAMES, reinv_prev)),
             "After re-investment and growth the field now holds %s flowers." % f1(pool_now),
             "This round the manager offers: " + "; ".join("%s %s" % (n, f1(v)) for n, v in zip(NAMES, offers_now)),
             "Only whole coins can be re-invested, so you can re-invest any whole number from 0 to %d of your %s flowers." % (mx, f1(o)),
             "How many do you re-invest in the field? Reply with one JSON object of the form {\"reinvest\": <integer>}. " + JSON_ONLY]
    return elicit.INSTR['flowers'] + "\n\n" + "\n".join(lines), mx
def matched_state(R_target, s, q, y, o_prev_each=None):
    """Construct a consistent previous round (equal previous offers) that yields pool ~R_target, then current offers with own share s."""
    # sum c_prev = o*(y+3q); R = min(200, 1.4*sum c)
    factor = 1.4 * (y + 3 * q)
    o = R_target / factor if o_prev_each is None else o_prev_each
    o = float(np.round(o))
    c_own = round(y * o); c_oth = [round(q * o)] * 3
    pool_prev = 4 * o
    pool_now = min(200.0, 1.4 * (c_own + sum(c_oth)))
    own = s * pool_now / 4; oth = (pool_now - own) / 3
    return dict(pool_prev=pool_prev, offers_prev=[o] * 4, reinv_prev=[c_own] + c_oth, pool_now=pool_now,
                offers_now=[own, oth, oth, oth], R=pool_now, s=s, q=c_oth[0] / o, y=c_own / o)
MATCHED = {'R_lo': (40, 1, .5, .5), 'R_hi': (140, 1, .5, .5), 's_lo': (70, .5, .5, .5), 's_hi': (70, 2, .5, .5),
           'q_lo': (70, 1, .2, .5), 'q_hi': (70, 1, .8, .5), 'y_lo': (70, 1, .5, .2), 'y_hi': (70, 1, .5, .8)}
# ---------------- fully neutral strategy questionnaire
NEUTRAL_INSTR = """DECISION TASK - INSTRUCTIONS
You are one of four members of a group. Your bonus payment depends on the points you accumulate, which depend on your own choices and on the choices of the other members.

The group shares a common fund that is administered by an allocator. The fund begins at 200 units and is capped at 200 units.
In every period the allocator distributes units from the fund to the four members. Allocations may differ across members and all allocations are visible to everyone.
Each member then chooses how many of the units allocated to them to return to the fund (whole units only). Every returned unit is multiplied by 1.4 and added to the fund, up to the cap of 200. Units that are not returned become the member's points.
All members choose simultaneously. A fund that reaches zero stays at zero, and no further allocations are possible.
After every period all members see the new size of the fund, every member's allocation and every member's return.

The number of periods is not announced. Your bonus is proportional to the points you earn from period 1 up to a period drawn at random, so you should behave as if the task might stop at any moment; points earned in later periods are less likely to count."""
def neutral_prompt():
    lines = ['ID | units in fund now | allocated to you | allocated to the other three members (total) | other three members returned in the previous period (% of their allocations) | you returned in the previous period (% of your allocation)']
    ids = []
    for k, (a, b, c, d, R, s, q, y) in enumerate(commons.cell_list()):
        mine = s * R / 4; others = R - mine; sid = 'S%02d' % (k + 1); ids.append(sid)
        lines.append('%s | %d | %d | %d | %d%% | %d%%' % (sid, round(R), round(mine), round(others), round(q * 100), round(y * 100)))
    ask = ("\n\nSTRATEGY QUESTIONNAIRE\nInstead of deciding period by period, tell us in advance what you would do in each situation below. "
           "Any of these situations can occur in the task, so treat each one as a real decision. For each situation give the percentage "
           "(a whole number from 0 to 100) of your current allocation that you would return to the fund.\n\n"
           "F: period 1 of the task. The fund holds 200 units and each of the four members is allocated 50 units.\n\n"
           "Situations in later periods:\n" + "\n".join(lines) +
           "\n\nReply with one JSON object that has the key \"F\" and the keys S01 to S81, each mapped to an integer from 0 to 100, for example {\"F\": 0, \"S01\": 0, ...}.")
    return NEUTRAL_INSTR + ask, ids
