#!/usr/bin/env python3
"""
Sistema analítico NBA: entrena modelos con logs reales de nba_api y pronostica
ganador, spread, total (over/under) y props de jugador.

Instalación:
    pip install nba_api pandas numpy scikit-learn scipy joblib

Uso:
    python nba_model.py train            # primera vez: descarga datos y entrena
    python nba_model.py update           # refresca datos de la temporada actual y reentrena
    python nba_model.py game BOS NYK --spread -4.5 --odds-spread-home -110 --odds-spread-away -110 \\
        --total 224.5 --odds-over -110 --odds-under -110 \\
        --total-home 114.5 --total-away 109.5
    python nba_model.py prop "Jayson Tatum" PTS 27.5 --opp NYK --home \
        --odds-over -115 --odds-under -105 --minutes 35

Para que se actualice solo, programa `update` una vez al día/semana, por ejemplo en cron:
    0 6 * * * cd /ruta/al/proyecto && python nba_model.py update

Props con tu propio dato: --avg 38.5 mezcla el promedio de esta temporada con el modelo (--peso-actual 0.5).

Registro de picks (para medir si el modelo funciona de verdad):
    python nba_model.py pick "Over 235.5" --odds -112 --prob 55.4
    python nba_model.py result 1 L        # W ganado, L perdido, P push
    python nba_model.py report

Ajustes útiles en `game`: --inj-home/--inj-away (bajas), --peso-modelo 0.3 (mezcla con la línea),
--ml-home/--ml-away (momios del ganador).

En `game` el primer equipo es el LOCAL. El spread es el del local (-4.5 = favorito).
Stats de props: PTS, REB, AST, FG3M, PRA (puntos + rebotes + asistencias).
"""
import argparse
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA, MODELS = Path("data"), Path("models")
# En un servidor (Streamlit Cloud) stats.nba.com suele bloquear la descarga: solo se leen los datos ya guardados.
NUBE = Path("/mount/src").exists() or os.environ.get("NBA_NUBE") == "1"
# Si en el repositorio los archivos quedaron sueltos (sin carpetas data/ y models/), se buscan en la raíz.
if NUBE and not DATA.exists() and Path("game.joblib").exists():
    DATA, MODELS = Path("."), Path(".")
STATS = ["ortg", "drtg", "pace", "efg", "tov", "orb", "ftr"]
TEAM_FEATS = ["r_" + s for s in STATS] + ["r_net", "rest"]
PSTATS = ["PTS", "REB", "AST", "FG3M", "PRA"]
HALF = 10  # vida media (partidos) del promedio exponencial: más peso a lo reciente


# ----------------------------------------------------------------- datos
def default_seasons(n=4):
    now = pd.Timestamp.today()
    y = now.year if now.month >= 10 else now.year - 1
    return [f"{k}-{str(k + 1)[-2:]}" for k in range(y - n + 1, y + 1)]


def fetch_logs(seasons, kind):
    """kind: 'T' (equipos) o 'P' (jugadores). Cachea en ./data y refresca la última temporada."""
    DATA.mkdir(exist_ok=True)
    frames = []
    for s in seasons:
        f = DATA / f"{kind}_{s}.csv"
        if NUBE or (f.exists() and s != seasons[-1]):
            if not f.exists():
                raise SystemExit(f"Falta {f}. Sube la carpeta 'data' completa al repositorio.")
            frames.append(pd.read_csv(f))
            continue
        try:
            from nba_api.stats.endpoints import leaguegamelog

            df = leaguegamelog.LeagueGameLog(
                season=s, player_or_team_abbreviation=kind, timeout=60
            ).get_data_frames()[0]
            df.to_csv(f, index=False)
            time.sleep(1.5)
        except Exception as e:  # sin conexión o bloqueo: se usa lo último guardado
            if not f.exists():
                raise
            print(f"Aviso: no se pudo actualizar {s} ({type(e).__name__}); se usan los datos guardados.")
            df = pd.read_csv(f)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"])
    return out.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


# ------------------------------------------------------ features de equipo
def team_games(logs):
    d = logs.copy()
    d["home"] = d["MATCHUP"].str.contains("vs.", regex=False).astype(int)
    d["poss"] = d.FGA + 0.44 * d.FTA - d.OREB + d.TOV
    opp = d[["GAME_ID", "TEAM_ID", "PTS", "poss", "DREB"]].rename(
        columns=lambda c: c if c == "GAME_ID" else "o_" + c
    )
    d = d.merge(opp, on="GAME_ID")
    d = d[d.TEAM_ID != d.o_TEAM_ID].copy()
    p = (d.poss + d.o_poss) / 2
    d["ortg"], d["drtg"], d["pace"] = 100 * d.PTS / p, 100 * d.o_PTS / p, p
    d["efg"] = (d.FGM + 0.5 * d.FG3M) / d.FGA
    d["tov"] = d.TOV / d.poss
    d["orb"] = d.OREB / (d.OREB + d.o_DREB)
    d["ftr"] = d.FTA / d.FGA
    d = d.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
    d["rest"] = d.groupby("TEAM_ID").GAME_DATE.diff().dt.days.fillna(7).clip(upper=7)
    return d


def add_rolling(d):
    """r_* = estado ANTES del partido (sin fuga de datos). c_* = estado incluyendo el último partido."""
    g = d.groupby("TEAM_ID")
    for s in STATS:
        d["r_" + s] = g[s].transform(lambda x: x.shift(1).ewm(halflife=HALF, min_periods=5).mean())
        d["c_" + s] = g[s].transform(lambda x: x.ewm(halflife=HALF, min_periods=5).mean())
    d["r_net"] = d.r_ortg - d.r_drtg
    d["c_net"] = d.c_ortg - d.c_drtg
    return d


def matchup_frame(d):
    h = d[d.home == 1].set_index("GAME_ID")[["GAME_DATE", "TEAM_ABBREVIATION", "PTS"] + TEAM_FEATS]
    a = d[d.home == 0].set_index("GAME_ID")[["TEAM_ABBREVIATION", "PTS"] + TEAM_FEATS]
    j = h.join(a, lsuffix="_h", rsuffix="_a", how="inner").dropna().reset_index()
    j = j.sort_values("GAME_DATE").reset_index(drop=True)
    j["margin"] = j.PTS_h - j.PTS_a
    j["total"] = j.PTS_h + j.PTS_a
    j["pts_h"], j["pts_a"] = j.PTS_h, j.PTS_a
    j["win"] = (j.margin > 0).astype(int)
    return j


def side(j, suf):
    return j[[f + suf for f in TEAM_FEATS]].rename(columns=lambda c: c[: -len(suf)])


def make_X(h, a):
    X = pd.DataFrame(index=range(len(h)))
    for s in STATS + ["net"]:
        X["h_" + s] = h["r_" + s].values
        X["a_" + s] = a["r_" + s].values
    X["d_net"] = X.h_net - X.a_net
    X["exp_pace"] = (X.h_pace + X.a_pace) / 2
    X["b2b_h"] = (h["rest"].values == 1).astype(int)
    X["b2b_a"] = (a["rest"].values == 1).astype(int)
    X["rest_d"] = h["rest"].values - a["rest"].values
    return X


def team_state(d, abbr, game_date):
    sub = d[d.TEAM_ABBREVIATION == abbr.upper()]
    if sub.empty:
        raise SystemExit(f"Equipo no encontrado: {abbr}")
    r = sub.sort_values("GAME_DATE").iloc[-1]
    s = {"r_" + k: r["c_" + k] for k in STATS + ["net"]}
    s["rest"] = min((pd.Timestamp(game_date) - r.GAME_DATE).days, 7)
    return pd.DataFrame([s])


# ------------------------------------------------------ features de jugador
def add_pra(pl):
    pl = pl.copy()
    pl["PRA"] = pl.PTS + pl.REB + pl.AST
    return pl


def rolled(p, shift):
    g, out = p.groupby("PLAYER_ID"), {}
    for s in PSTATS + ["MIN"]:
        out["e_" + s] = g[s].transform(lambda x: x.shift(shift).ewm(halflife=HALF, min_periods=5).mean())
        out["l5_" + s] = g[s].transform(lambda x: x.shift(shift).rolling(5, min_periods=3).mean())
        out["s_" + s] = g[s].transform(lambda x: x.shift(shift).expanding(min_periods=5).mean())
    return pd.DataFrame(out, index=p.index)


def player_frame(pl, d):
    p = add_pra(pl)
    p["home"] = p.MATCHUP.str.contains("vs.", regex=False).astype(int)
    p["opp"] = p.MATCHUP.str[-3:]
    p = p.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    p["rest"] = p.groupby("PLAYER_ID").GAME_DATE.diff().dt.days.fillna(7).clip(upper=7)
    t = d[["GAME_ID", "TEAM_ABBREVIATION", "r_drtg", "r_pace"]]
    p = p.merge(
        t.rename(columns={"TEAM_ABBREVIATION": "opp", "r_drtg": "o_drtg", "r_pace": "o_pace"}),
        on=["GAME_ID", "opp"], how="left",
    )
    p = p.merge(
        t.rename(columns={"r_drtg": "t_drtg", "r_pace": "t_pace"}),
        on=["GAME_ID", "TEAM_ABBREVIATION"], how="left",
    )
    p["exp_pace"] = (p.o_pace + p.t_pace) / 2
    return p.join(rolled(p, 1))


def prop_cols(s):
    return [f"e_{s}", f"l5_{s}", f"s_{s}", "e_MIN", "l5_MIN", "o_drtg", "exp_pace", "home", "rest"]


# ----------------------------------------------------------------- entrenar
def train(seasons, half=10):
    global HALF
    HALF = half
    MODELS.mkdir(exist_ok=True)
    logs = fetch_logs(seasons, "T")
    d = add_rolling(team_games(logs))
    j = matchup_frame(d)
    X = make_X(side(j, "_h"), side(j, "_a"))
    cut = int(len(j) * 0.8)  # validación temporal: se entrena con el pasado, se prueba con el futuro
    clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=1000))
    reg = lambda: make_pipeline(StandardScaler(), Ridge(alpha=200))

    c = clf().fit(X.iloc[:cut], j.win.iloc[:cut])
    p = c.predict_proba(X.iloc[cut:])[:, 1]
    yt, base = j.win.iloc[cut:], j.win.iloc[:cut].mean()
    print(f"[Ganador] partidos prueba: {len(yt)}")
    print(f"  acierto {accuracy_score(yt, p > .5):.3f} | brier {brier_score_loss(yt, p):.4f} "
          f"(base {brier_score_loss(yt, np.full(len(yt), base)):.4f}) | logloss {log_loss(yt, p):.4f}")

    bundle = {"seasons": seasons, "half": half}
    for name in ("margin", "total", "pts_h", "pts_a"):
        m = reg().fit(X.iloc[:cut], j[name].iloc[:cut])
        pr = m.predict(X.iloc[cut:])
        y = j[name].iloc[cut:]
        mae_b = mean_absolute_error(y, np.full(len(y), j[name].iloc[:cut].mean()))
        sig = float(np.std(y - pr))
        print(f"[{name}] MAE {mean_absolute_error(y, pr):.2f} (base {mae_b:.2f}) | desv. residual {sig:.2f}")
        bundle[name] = reg().fit(X, j[name])
        bundle["sig_" + name] = sig
    bundle["clf"] = clf().fit(X, j.win)
    joblib.dump(bundle, MODELS / "game.joblib")

    pl = fetch_logs(seasons, "P")
    p = player_frame(pl, d)
    props = {"seasons": seasons, "half": half}
    for s in PSTATS:
        cols = prop_cols(s)
        q = p.dropna(subset=cols + [s])
        q = q[q.e_MIN >= 15].sort_values("GAME_DATE").reset_index(drop=True)
        k = int(len(q) * 0.8)
        m = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(q[cols].iloc[:k], q[s].iloc[:k])
        pr = m.predict(q[cols].iloc[k:])
        y = q[s].iloc[k:].values
        res = y - pr
        coef = np.polyfit(pr, np.abs(res) * 1.2533, 1)  # desviación proporcional al nivel esperado
        base = mean_absolute_error(y, q["s_" + s].iloc[k:])
        print(f"[{s}] MAE {mean_absolute_error(y, pr):.2f} (promedio temporada {base:.2f})")
        props[s] = {"model": m.fit(q[cols], q[s]), "cols": cols, "sd": coef}
    joblib.dump(props, MODELS / "props.joblib")
    print(f"Modelos guardados en ./models (vida media = {half} partidos)")


# ------------------------------------------------------------- predicción
def imp(o):
    return -o / (-o + 100) if o < 0 else 100 / (o + 100)


def show(label, p, odds):
    m = imp(odds)
    e = p - m
    tag = "CASHHH" if e >= 0.03 else "-"
    print(f"  {label:<26} modelo {p:6.1%} | cuota {odds:+d} ({m:5.1%}) | ventaja {e * 100:+5.1f} pts  {tag}")


SIG_M, SIG_T = 12.5, 18.0  # desviaciones típicas realistas de margen y total en la NBA
SIG_P = ((SIG_M ** 2 + SIG_T ** 2) / 4) ** 0.5  # desviación de los puntos de un solo equipo


def over_under(title, mean, sigma, line, odds_o, odds_u):
    po = 1 - norm.cdf((line - mean) / sigma)
    print(f"{title} (proyección {mean:.1f})")
    show(f"Over {line:g}", po, odds_o)
    show(f"Under {line:g}", 1 - po, odds_u)


def cmd_game(a):
    global HALF
    b = joblib.load(MODELS / "game.joblib")
    HALF = b.get("half", HALF)
    d = add_rolling(team_games(fetch_logs(b["seasons"], "T")))
    date = pd.Timestamp(a.date) if a.date else pd.Timestamp.today().normalize()
    h, v = team_state(d, a.home, date), team_state(d, a.away, date)
    X = make_X(h, v)
    H, V = a.home.upper(), a.away.upper()
    mar0, tot0 = b["margin"].predict(X)[0], b["total"].predict(X)[0]
    # 1) bajas: restan net rating al equipo afectado (el total baja un poco)
    mar = mar0 - a.inj_home + a.inj_away
    tot = tot0 - 0.5 * (a.inj_home + a.inj_away)
    # 2) el mercado es más preciso que el modelo: se mezcla con la línea de la casa
    w = a.peso_modelo
    if a.spread is not None:
        mar = w * mar + (1 - w) * (-a.spread)
    if a.total is not None:
        tot = w * tot + (1 - w) * a.total
    ph, pa = (tot + mar) / 2, (tot - mar) / 2
    pw = norm.cdf(mar / SIG_M)
    print(f"\n{H} (local) vs {V} (visitante)")
    print(f"  Modelo puro: margen {mar0:+.1f} | total {tot0:.1f}")
    print(f"  Ajustado (bajas + {1 - w:.0%} mercado): margen {mar:+.1f} | total {tot:.1f}")
    print(f"  Prob. victoria: {H} {pw:.1%} | {V} {1 - pw:.1%}")
    print(f"  Marcador proyectado: {H} {ph:.0f} - {V} {pa:.0f}\n")
    if a.ml_home is not None or a.ml_away is not None:
        print("Ganador")
        if a.ml_home is not None:
            show(f"Local {H}", pw, a.ml_home)
        if a.ml_away is not None:
            show(f"Visitante {V}", 1 - pw, a.ml_away)
        print()
    if a.spread is not None:
        pc = norm.cdf((mar + a.spread) / SIG_M)
        print("Spread")
        show(f"Local {H} {a.spread:+g}", pc, a.odds_spread_home)
        show(f"Visitante {V} {-a.spread:+g}", 1 - pc, a.odds_spread_away)
        print()
    if a.total is not None:
        over_under("Total del partido", tot, SIG_T, a.total, a.odds_over, a.odds_under)
        print()
    if a.total_home is not None:
        over_under(f"Puntos del local {H}", ph, SIG_P, a.total_home, a.odds_home_over, a.odds_home_under)
        print()
    if a.total_away is not None:
        over_under(f"Puntos del visitante {V}", pa, SIG_P, a.total_away, a.odds_away_over, a.odds_away_under)


def cmd_prop(a):
    global HALF
    b = joblib.load(MODELS / "props.joblib")
    HALF = b.get("half", HALF)
    stat = a.stat.upper()
    if stat not in PSTATS:
        raise SystemExit(f"Stat válida: {', '.join(PSTATS)}")
    d = add_rolling(team_games(fetch_logs(b["seasons"], "T")))
    pl = add_pra(fetch_logs(b["seasons"], "P"))
    m = pl[pl.PLAYER_NAME.str.contains(a.player, case=False, regex=False)]
    if m.PLAYER_ID.nunique() != 1:
        raise SystemExit(f"Jugadores encontrados: {sorted(m.PLAYER_NAME.unique())[:8]}")
    p = m.sort_values("GAME_DATE").reset_index(drop=True)
    r = rolled(p, 0).iloc[-1].to_dict()
    date = pd.Timestamp.today().normalize()
    own, opp = p.TEAM_ABBREVIATION.iloc[-1], team_state(d, a.opp, date)
    r.update(o_drtg=opp.r_drtg[0], exp_pace=(opp.r_pace[0] + team_state(d, own, date).r_pace[0]) / 2,
             home=int(a.home), rest=a.rest)
    info = b[stat]
    pred = info["model"].predict(pd.DataFrame([r])[info["cols"]])[0]
    mins_note = f"minutos de tendencia {r['e_MIN']:.1f}"
    if a.minutes:  # rendimiento por minuto x minutos que esperas que juegue
        pred *= a.minutes / max(r["e_MIN"], 1.0)
        mins_note += f" -> esperados {a.minutes:g}"
    if a.avg is not None:  # tu dato: promedio de esta temporada, mezclado con la proyección del modelo
        pred = a.peso_actual * a.avg + (1 - a.peso_actual) * pred
        mins_note += f" | mezcla {a.peso_actual:.0%} con tu promedio {a.avg:g}"
    sd = max(np.polyval(info["sd"], pred), 0.5)
    po = 1 - norm.cdf((a.line - pred) / sd)
    print(f"\n{p.PLAYER_NAME.iloc[-1]} ({own}) vs {a.opp.upper()} | {stat}")
    print(f"  Proyección {pred:.1f} | desviación {sd:.1f} | línea {a.line:g} | {mins_note}")
    show(f"Over {a.line:g}", po, a.odds_over)
    show(f"Under {a.line:g}", 1 - po, a.odds_under)


# ------------------------------------------------------- registro de picks
PICKS = Path("picks.csv")
PICK_COLS = ["id", "fecha", "pick", "cuota", "prob_modelo", "resultado"]


def load_picks():
    if PICKS.exists():
        return pd.read_csv(PICKS, dtype={"resultado": str}, keep_default_na=False)
    return pd.DataFrame(columns=PICK_COLS)


def cmd_pick(a):
    df = load_picks()
    nid = int(df.id.max()) + 1 if len(df) else 1
    df.loc[len(df)] = [nid, str(pd.Timestamp.today().date()), a.name, a.odds, a.prob / 100, ""]
    df.to_csv(PICKS, index=False)
    print(f"Pick #{nid} guardado: {a.name} ({a.odds:+d}). Cuando termine el partido: result {nid} W|L|P")


def cmd_result(a):
    df = load_picks()
    if a.id not in df.id.astype(int).values:
        raise SystemExit(f"No existe el pick #{a.id}")
    df.loc[df.id.astype(int) == a.id, "resultado"] = a.res.upper()
    df.to_csv(PICKS, index=False)
    print(f"Pick #{a.id} marcado como {a.res.upper()}")


def cmd_report(a):
    df = load_picks()
    pend = df[~df.resultado.isin(["W", "L", "P"])]
    d = df[df.resultado.isin(["W", "L", "P"])].copy()
    if len(pend):
        print("Pendientes: " + ", ".join(f"#{r.id} {r.pick}" for r in pend.itertuples()))
    if d.empty:
        print("Aún no hay picks con resultado.")
        return

    def profit(r):  # 1 unidad por pick
        if r.resultado == "P":
            return 0.0
        if r.resultado == "L":
            return -1.0
        return 100 / -r.cuota if r.cuota < 0 else r.cuota / 100

    d["u"] = d.apply(profit, axis=1)
    dec = d[d.resultado != "P"]
    w, l = int((dec.resultado == "W").sum()), int((dec.resultado == "L").sum())
    print(f"Récord {w}-{l} | unidades {d.u.sum():+.2f} | ROI {d.u.sum() / len(d):+.1%}")
    if len(dec):
        print(f"Prob. media del modelo {dec.prob_modelo.astype(float).mean():.1%} | "
              f"acierto real {w / len(dec):.1%} | implícita media de las cuotas {dec.cuota.astype(int).map(imp).mean():.1%}")
    if len(dec) < 30:
        print("Aviso: con menos de ~30 picks la muestra es muy pequeña para sacar conclusiones.")


# --------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--seasons", nargs="+", default=default_seasons())
    t.add_argument("--half", type=int, default=10, help="vida media (partidos) de las estadísticas recientes")
    u = sp.add_parser("update")
    u.add_argument("--half", type=int, default=10)
    k = sp.add_parser("pick")
    k.add_argument("name", help='ej. "GSW -3.5" o "Over 235.5"')
    k.add_argument("--odds", type=int, required=True)
    k.add_argument("--prob", type=float, required=True, help="prob. del modelo en %%, ej. 55.4")
    rs = sp.add_parser("result")
    rs.add_argument("id", type=int)
    rs.add_argument("res", choices=["W", "L", "P", "w", "l", "p"], help="W ganado, L perdido, P push")
    sp.add_parser("report")
    g = sp.add_parser("game")
    g.add_argument("home")
    g.add_argument("away")
    g.add_argument("--spread", type=float, help="spread del local, ej. -4.5")
    g.add_argument("--ml-home", type=int, help="momio ganador local")
    g.add_argument("--ml-away", type=int, help="momio ganador visitante")
    g.add_argument("--inj-home", type=float, default=0, help="net rating perdido por bajas del local (estrella 3-5, rotación 1-2)")
    g.add_argument("--inj-away", type=float, default=0, help="net rating perdido por bajas del visitante")
    g.add_argument("--peso-modelo", type=float, default=0.3, help="peso del modelo vs la línea de la casa (0-1)")
    g.add_argument("--odds-spread-home", type=int, default=-110)
    g.add_argument("--odds-spread-away", type=int, default=-110)
    g.add_argument("--total", type=float, help="total del partido")
    g.add_argument("--odds-over", type=int, default=-110)
    g.add_argument("--odds-under", type=int, default=-110)
    g.add_argument("--total-home", type=float, help="puntos del equipo local")
    g.add_argument("--odds-home-over", type=int, default=-110)
    g.add_argument("--odds-home-under", type=int, default=-110)
    g.add_argument("--total-away", type=float, help="puntos del equipo visitante")
    g.add_argument("--odds-away-over", type=int, default=-110)
    g.add_argument("--odds-away-under", type=int, default=-110)
    g.add_argument("--date", help="YYYY-MM-DD del partido (por defecto hoy)")
    r = sp.add_parser("prop")
    r.add_argument("player")
    r.add_argument("stat")
    r.add_argument("line", type=float)
    r.add_argument("--opp", required=True)
    r.add_argument("--home", action="store_true")
    r.add_argument("--minutes", type=float, help="minutos esperados (por defecto, tendencia reciente)")
    r.add_argument("--avg", type=float, help="promedio de esta temporada del jugador en esa estadística (dato tuyo)")
    r.add_argument("--peso-actual", type=float, default=0.5, help="peso de --avg frente al modelo (0-1)")
    r.add_argument("--rest", type=int, default=2, help="días de descanso")
    r.add_argument("--odds-over", type=int, default=-110)
    r.add_argument("--odds-under", type=int, default=-110)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.seasons, a.half)
    elif a.cmd == "update":
        train(default_seasons(), a.half)
    elif a.cmd == "pick":
        cmd_pick(a)
    elif a.cmd == "result":
        cmd_result(a)
    elif a.cmd == "report":
        cmd_report(a)
    elif a.cmd == "game":
        cmd_game(a)
    else:
        cmd_prop(a)


if __name__ == "__main__":
    main()
