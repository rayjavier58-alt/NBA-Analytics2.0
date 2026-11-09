#!/usr/bin/env python3
"""
Interfaz del modelo NBA (Streamlit). Va en la MISMA carpeta que nba_model.py.

    pip install streamlit
    streamlit run app.py

Se abre sola en el navegador (http://localhost:8501).
"""
import base64
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

import nba_model as m

PICKS_APP = Path("picks_app.csv")
PICK_COLS = ["fecha", "pick", "momio", "monto", "resultado"]
RESULTADOS = ["Pendiente", "Ganado", "Perdido", "Push"]
CASHHH_MIN = 0.03

EQUIPOS = {
    "ATL": ("Atlanta Hawks", "atl"), "BOS": ("Boston Celtics", "bos"), "BKN": ("Brooklyn Nets", "bkn"),
    "CHA": ("Charlotte Hornets", "cha"), "CHI": ("Chicago Bulls", "chi"), "CLE": ("Cleveland Cavaliers", "cle"),
    "DAL": ("Dallas Mavericks", "dal"), "DEN": ("Denver Nuggets", "den"), "DET": ("Detroit Pistons", "det"),
    "GSW": ("Golden State Warriors", "gs"), "HOU": ("Houston Rockets", "hou"), "IND": ("Indiana Pacers", "ind"),
    "LAC": ("Los Angeles Clippers", "lac"), "LAL": ("Los Angeles Lakers", "lal"), "MEM": ("Memphis Grizzlies", "mem"),
    "MIA": ("Miami Heat", "mia"), "MIL": ("Milwaukee Bucks", "mil"), "MIN": ("Minnesota Timberwolves", "min"),
    "NOP": ("New Orleans Pelicans", "no"), "NYK": ("New York Knicks", "ny"), "OKC": ("Oklahoma City Thunder", "okc"),
    "ORL": ("Orlando Magic", "orl"), "PHI": ("Philadelphia 76ers", "phi"), "PHX": ("Phoenix Suns", "phx"),
    "POR": ("Portland Trail Blazers", "por"), "SAC": ("Sacramento Kings", "sac"), "SAS": ("San Antonio Spurs", "sa"),
    "TOR": ("Toronto Raptors", "tor"), "UTA": ("Utah Jazz", "utah"), "WAS": ("Washington Wizards", "wsh"),
}


def nombre(abbr):
    return EQUIPOS.get(abbr, (abbr, ""))[0]


def logo(abbr):
    cod = EQUIPOS.get(abbr, ("", abbr.lower()))[1]
    return f"https://a.espncdn.com/i/teamlogos/nba/500/{cod}.png"


# ------------------------------------------------------------ lógica (sin UI)
def fila(mercado, p, odds):
    imp = m.imp(odds)
    e = p - imp
    return {"Mercado": mercado, "Modelo": p, "Cuota": int(odds), "Implícita": imp,
            "Ventaja (pts)": e * 100, "Señal": "CASHHH" if e >= CASHHH_MIN else "-"}


def cargar_modelo():
    b = joblib.load(m.MODELS / "game.joblib")
    m.HALF = b.get("half", m.HALF)
    d = m.add_rolling(m.team_games(m.fetch_logs(b["seasons"], "T")))
    return b, d


def _norm(x):
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", x).strip()


def cargar_jugadores(b):
    pl = m.add_pra(m.fetch_logs(b["seasons"], "P"))
    pl["NOMBRE_N"] = pl.PLAYER_NAME.map(_norm)
    return pl


# ------------------------------------------------------------- bajas automáticas
ESPN_BAJAS = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
PESO_ESTADO = (("out", 1.0), ("doubtful", 0.8), ("question", 0.5), ("day", 0.5), ("probable", 0.15))


def parsear_bajas(data):
    """{abreviatura: [(jugador, estado, comentario)]} a partir del JSON de lesiones de ESPN."""
    por_equipo = {}
    for t in data.get("injuries", []):
        ultimo = str(t.get("displayName", "")).split()[-1].lower() if t.get("displayName") else ""
        for ab, (nom, _) in EQUIPOS.items():
            if nom.split()[-1].lower() == ultimo:
                for i in t.get("injuries", []):
                    jug = (i.get("athlete") or {}).get("displayName")
                    if jug:
                        por_equipo.setdefault(ab, []).append((jug, i.get("status", ""), i.get("shortComment", "")))
    return por_equipo


def bajas_espn():
    import requests

    r = requests.get(ESPN_BAJAS, timeout=15)
    r.raise_for_status()
    return parsear_bajas(r.json())


def total_bajas(df, juegan=()):
    """Suma de bajas. Un jugador marcado como 'juega' deja de restar y, si venía faltando varios partidos,
    suma un refuerzo (valor negativo) porque las estadísticas recientes del equipo ya reflejan su ausencia."""
    if df is None or df.empty:
        return 0.0
    tot = 0.0
    for _, r in df.iterrows():
        tot += -r["Impacto si falta"] * r["Ya reflejado"] if r["Jugador"] in juegan else r["Aporte"]
    return max(min(float(tot), 12.0), -12.0)


def localia_auto(d, home, away):
    """Ajuste automático (puntos de margen) de la ventaja de localía: la de la liga en el último año frente a la
    media del entrenamiento, más una pequeña corrección por equipo (muy encogida: la muestra es corta)."""
    x = d.copy()
    x["mar"] = x.PTS - x.o_PTS
    rec = x[x.GAME_DATE >= x.GAME_DATE.max() - pd.Timedelta(days=365)]
    liga_rec = float(rec[rec.home == 1].mar.mean())
    liga_ent = float(x[x.home == 1].mar.mean())

    def ha(df):
        h, a = df[df.home == 1].mar, df[df.home == 0].mar
        return (h.mean() - a.mean()) / 2 if len(h) and len(a) else liga_rec

    def dev(t):
        s_ = rec[rec.TEAM_ABBREVIATION == t]
        return len(s_) / (len(s_) + 150) * (ha(s_) - liga_rec)

    eq = dev(home) - dev(away)
    tot = float(max(min(liga_rec - liga_ent + eq, 3.0), -3.0))
    return tot, (f"Localía de la liga en el último año: {liga_rec:+.1f} pts (entrenamiento {liga_ent:+.1f}); "
                 f"ajuste por equipos {eq:+.1f}.")


def estimar_bajas(pl, d, abbr, lista):
    """Impacto estimado (net rating perdido) de cada baja. Heurística propia, no un dato oficial:
    sale de la producción reciente (puntos + 0.7 asist. + 0.4 reb.) y se descuenta si el jugador
    ya lleva partidos fuera, porque entonces el modelo ya lo tiene en parte reflejado."""
    cols = ["Jugador", "Estado", "Prob. no juega", "Impacto si falta", "Ya reflejado", "Aporte"]
    pj = d[d.TEAM_ABBREVIATION == abbr].sort_values("GAME_DATE")
    ult = set(pj.GAME_ID.tail(5))
    plantilla = pl[(pl.TEAM_ABBREVIATION == abbr) & pl.GAME_ID.isin(set(pj.GAME_ID.tail(15)))]
    filas = []
    for nombre_j, estado, _ in lista:
        n = _norm(nombre_j)
        cand = pl[pl.NOMBRE_N == n]
        if cand.empty and n:
            cand = plantilla[plantilla.NOMBRE_N.str.split().str[-1] == n.split()[-1]]
            if cand.PLAYER_ID.nunique() != 1:
                cand = cand.iloc[0:0]
        w = next((v for k, v in PESO_ESTADO if k in str(estado).lower()), 0.5)
        if cand.empty:
            filas.append([nombre_j, estado, w, 0.0, 0.0, 0.0])
            continue
        pid = cand.sort_values("GAME_DATE").PLAYER_ID.iloc[-1]
        c = pl[pl.PLAYER_ID == pid].sort_values("GAME_DATE").tail(10)
        base = max(0.0, (c.PTS + 0.7 * c.AST + 0.4 * c.REB).mean() - 12) * 0.16
        faltas = len(ult) - len(set(pl[(pl.PLAYER_ID == pid) & pl.GAME_ID.isin(ult)].GAME_ID))
        refl = 0.75 * faltas / max(len(ult), 1)
        filas.append([nombre_j, estado, w, base, refl, base * w * (1 - refl)])
    return pd.DataFrame(filas, columns=cols)


def analizar_partido(b, d, home, away, *, spread=None, o_sh=-110, o_sa=-110, total=None, o_over=-110,
                     o_under=-110, ml_h=None, ml_a=None, tot_h=None, o_hover=-110, o_hunder=-110,
                     tot_a=None, o_aover=-110, o_aunder=-110, inj_h=0.0, inj_a=0.0, peso=0.3, localia=0.0, fecha=None):
    fecha = pd.Timestamp(fecha) if fecha else pd.Timestamp.today().normalize()
    X = m.make_X(m.team_state(d, home, fecha), m.team_state(d, away, fecha))
    mar0, tot0 = float(b["margin"].predict(X)[0]), float(b["total"].predict(X)[0])
    mar, tot = mar0 + localia - inj_h + inj_a, tot0 - 0.5 * (inj_h + inj_a)
    if spread is not None:
        mar = peso * mar + (1 - peso) * (-spread)
    if total is not None:
        tot = peso * tot + (1 - peso) * total
    ph, pa = (tot + mar) / 2, (tot - mar) / 2
    pw = float(norm.cdf(mar / m.SIG_M))
    H, V = home.upper(), away.upper()
    filas = []
    if ml_h is not None:
        filas.append(fila(f"Ganador {H}", pw, ml_h))
    if ml_a is not None:
        filas.append(fila(f"Ganador {V}", 1 - pw, ml_a))
    if spread is not None:
        pc = float(norm.cdf((mar + spread) / m.SIG_M))
        filas += [fila(f"{H} {spread:+g}", pc, o_sh), fila(f"{V} {-spread:+g}", 1 - pc, o_sa)]

    def ou(nombre, media, sig, linea, oo, ou_):
        po = float(1 - norm.cdf((linea - media) / sig))
        return [fila(f"{nombre} Over {linea:g}", po, oo), fila(f"{nombre} Under {linea:g}", 1 - po, ou_)]

    if total is not None:
        filas += ou("Total", tot, m.SIG_T, total, o_over, o_under)
    if tot_h is not None:
        filas += ou(f"Puntos {H}", ph, m.SIG_P, tot_h, o_hover, o_hunder)
    if tot_a is not None:
        filas += ou(f"Puntos {V}", pa, m.SIG_P, tot_a, o_aover, o_aunder)
    return {"home": H, "away": V, "mar0": mar0, "tot0": tot0, "localia": localia, "mar": mar, "tot": tot, "ph": ph,
            "pa": pa, "pw": pw, "tabla": pd.DataFrame(filas)}


def analizar_prop(b, d, pl, nombre, stat, linea, o_over, o_under, opp, home, *, minutos=None,
                  promedio=None, peso_actual=0.5, descanso=2, fecha=None):
    props = joblib.load(m.MODELS / "props.joblib")
    m.HALF = props.get("half", m.HALF)
    fecha = pd.Timestamp(fecha) if fecha else pd.Timestamp.today().normalize()
    p = pl[pl.PLAYER_NAME == nombre].sort_values("GAME_DATE").reset_index(drop=True)
    r = m.rolled(p, 0).iloc[-1].to_dict()
    propio = p.TEAM_ABBREVIATION.iloc[-1]
    to, tt = m.team_state(d, opp, fecha), m.team_state(d, propio, fecha)
    r.update(o_drtg=to.r_drtg[0], exp_pace=(to.r_pace[0] + tt.r_pace[0]) / 2, home=int(home), rest=descanso)
    info = props[stat]
    pred = float(info["model"].predict(pd.DataFrame([r])[info["cols"]])[0])
    modelo_puro = pred
    if minutos:
        pred *= minutos / max(r["e_MIN"], 1.0)
    if promedio is not None:
        pred = peso_actual * promedio + (1 - peso_actual) * pred
    sd = max(float(np.polyval(info["sd"], pred)), 0.5)
    po = float(1 - norm.cdf((linea - pred) / sd))
    tabla = pd.DataFrame([fila(f"Over {linea:g}", po, o_over), fila(f"Under {linea:g}", 1 - po, o_under)])
    return {"equipo": propio, "pred": pred, "puro": modelo_puro, "sd": sd, "min_tend": r["e_MIN"], "tabla": tabla}


def ganancia(r):
    try:
        mo, mt = float(r["momio"]), float(r["monto"])
    except (TypeError, ValueError):
        return np.nan
    if r["resultado"] == "Ganado":
        return mt * 100 / -mo if mo < 0 else mt * mo / 100
    if r["resultado"] == "Perdido":
        return -mt
    if r["resultado"] == "Push":
        return 0.0
    return np.nan


def cargar_picks():
    if PICKS_APP.exists():
        df = pd.read_csv(PICKS_APP, dtype={"resultado": str, "pick": str}, keep_default_na=False)
    else:
        df = pd.DataFrame(columns=PICK_COLS)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
    df["resultado"] = df["resultado"].replace("", "Pendiente")
    for c in ("momio", "monto"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[PICK_COLS]


def guardar_picks(df):
    df = df.copy()
    df["fecha"] = df["fecha"].fillna(pd.Timestamp.today().date())
    df["resultado"] = df["resultado"].fillna("Pendiente")
    df.to_csv(PICKS_APP, index=False)


def agregar_pick(pick, momio, monto):
    df = cargar_picks()
    df.loc[len(df)] = [pd.Timestamp.today().date(), pick, momio, monto, "Pendiente"]
    guardar_picks(df)


def resumen(df, dia):
    d = df.copy()
    d["gan"] = d.apply(ganancia, axis=1)
    res = d[d.resultado.isin(["Ganado", "Perdido", "Push"])]
    deldia = res[res.fecha == dia]
    dec = lambda x: x[x.resultado.isin(["Ganado", "Perdido"])]
    return {
        "dia_bal": float(deldia.gan.sum()), "dia_n": int((d.fecha == dia).sum()),
        "dia_apostado": float(dec(deldia).monto.sum()),
        "tot_bal": float(res.gan.sum()), "tot_apostado": float(dec(res).monto.sum()),
        "ganados": int((res.resultado == "Ganado").sum()), "perdidos": int((res.resultado == "Perdido").sum()),
        "pendiente": float(d[d.resultado == "Pendiente"].monto.sum()),
    }



CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&display=swap');
.block-container{padding-top:1.4rem;max-width:1200px}
.topbar{display:flex;align-items:flex-end;justify-content:space-between;gap:10px;overflow:hidden;
 background:linear-gradient(180deg,#0a1a33,#122e57);border-radius:14px;padding:10px 16px 0;margin-bottom:10px;border-bottom:4px solid #c9a45c}
.topbar .side{width:20%;max-width:210px;display:block}
.titulo{flex:1;display:flex;align-items:center;justify-content:center;gap:16px;padding:14px 0 18px;text-align:center}
.titulo .nba{font-family:Anton,Impact,'Arial Narrow',sans-serif;font-size:clamp(34px,6.5vw,76px);letter-spacing:.06em;
 line-height:1;text-transform:uppercase;background:linear-gradient(180deg,#f6e7b8,#c9a45c 55%,#8a6a2b);
 -webkit-background-clip:text;background-clip:text;color:transparent}
.titulo .sub{margin-top:6px;font-size:clamp(10px,1.4vw,15px);letter-spacing:.35em;text-transform:uppercase;color:#c9a45c}
.titulo .trofeo{height:clamp(54px,9vw,112px);width:auto;font-size:clamp(46px,8vw,96px);line-height:1}
@media(max-width:700px){.topbar .side{display:none}}
.match{display:grid;grid-template-columns:1fr 2fr 1fr;align-items:center;gap:12px;background:#0f1d33;color:#fff;
 border-radius:14px;padding:18px;margin:10px 0 14px}
.team{text-align:center}.team img{width:84px;height:84px;object-fit:contain}
.tn{font-weight:700;margin-top:6px}.loc{font-size:12px;opacity:.7}
.mid{text-align:center}.probs{display:flex;justify-content:center;align-items:baseline;gap:14px}
.p1,.p2{font-size:38px;font-weight:800}.p1{color:#7fb0ff}.p2{color:#f2b53d}.vs{opacity:.6}
.bar{height:10px;background:#f2b53d;border-radius:6px;overflow:hidden;margin:8px 0}
.bar div{height:100%;background:#7fb0ff}
.score{font-size:22px;font-weight:700}.tot{font-size:13px;opacity:.75}
[data-testid="stMetric"]{background:rgba(127,127,127,.08);border:1px solid rgba(127,127,127,.2);
 padding:12px 14px;border-radius:12px}
@media(max-width:700px){.match{grid-template-columns:1fr}}
</style>
"""

def _foto(nombre):
    for ext, mime in (("jpg", "jpeg"), ("jpeg", "jpeg"), ("png", "png"), ("webp", "webp")):
        p = Path(f"{nombre}.{ext}")
        if p.exists():
            return f"data:image/{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    return None


def _figura(cx, nombre, etiqueta, color, fg, numero):
    uri = _foto(nombre)
    if uri:  # tu foto (mejor PNG con fondo transparente)
        cuerpo = f'<image href="{uri}" x="{cx - 95}" y="205" width="190" height="235" preserveAspectRatio="xMidYMax meet"/>'
    else:  # silueta genérica con la camiseta
        c = "#39405a"
        cuerpo = (
            f'<circle cx="{cx}" cy="262" r="15" fill="{c}"/>'
            f'<polygon points="{cx - 30},282 {cx + 30},282 {cx + 26},352 {cx - 26},352" fill="{color}"/>'
            f'<polygon points="{cx - 30},284 {cx - 46},340 {cx - 36},346 {cx - 22},294" fill="{c}"/>'
            f'<polygon points="{cx + 30},284 {cx + 46},340 {cx + 36},346 {cx + 22},294" fill="{c}"/>'
            f'<polygon points="{cx - 26},352 {cx + 26},352 {cx + 31},394 {cx - 31},394" fill="{color}" opacity=".85"/>'
            f'<rect x="{cx - 26}" y="394" width="20" height="36" fill="{c}"/><rect x="{cx + 6}" y="394" width="20" height="36" fill="{c}"/>'
            f'<rect x="{cx - 32}" y="430" width="28" height="8" rx="3" fill="#111"/><rect x="{cx + 4}" y="430" width="28" height="8" rx="3" fill="#111"/>'
            f'<text x="{cx}" y="332" font-size="30" font-weight="900" text-anchor="middle" fill="{fg}" '
            f'font-family="Impact,Arial Black,sans-serif">{numero}</text>')
    placa = (f'<rect x="{cx - 62}" y="414" width="124" height="18" rx="9" fill="#000" opacity=".65"/>'
             f'<text x="{cx}" y="427" font-size="11" font-weight="700" text-anchor="middle" fill="#fff" '
             f'font-family="Arial,sans-serif">{etiqueta}</text>')
    return cuerpo + placa


def _lado(cx, nombre, etiqueta, color, fg, numero):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{cx - 100} 205 200 235">'
           f'{_figura(cx, nombre, etiqueta, color, fg, numero)}</svg>')
    return f'<img class="side" alt="" src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}">'


def hero_html():
    """Banner: imagen a la izquierda, título al centro (con trofeo) e imagen a la derecha.
    Fotos opcionales junto a app.py: curry.png, lamelo.png y trofeo.png (también .jpg/.webp)."""
    izq = _lado(103, "curry", "STEPHEN CURRY · 30", "#1d428a", "#ffc72c", 30)
    der = _lado(1097, "lamelo", "LAMELO BALL · 1", "#00788c", "#ffffff", 1)
    t = _foto("trofeo")
    trofeo = f'<img class="trofeo" alt="" src="{t}">' if t else '<span class="trofeo">🏆</span>'
    return (f'<div class="topbar">{izq}<div class="titulo">{trofeo}<div><div class="nba">NBA Analytics</div>'
            f'<div class="sub">Ganador · Spread · Totales · Props</div></div></div>{der}</div>')


def tarjeta_partido(r):
    ph = r["pw"]
    return f"""
<div class="match">
  <div class="team"><img src="{logo(r['home'])}"><div class="tn">{nombre(r['home'])}</div><div class="loc">Local</div></div>
  <div class="mid">
    <div class="probs"><span class="p1">{ph:.0%}</span><span class="vs">vs</span><span class="p2">{1 - ph:.0%}</span></div>
    <div class="bar"><div style="width:{ph * 100:.1f}%"></div></div>
    <div class="score">{r['ph']:.0f} – {r['pa']:.0f}</div>
    <div class="tot">Total proyectado {r['tot']:.1f} · Margen local {r['mar']:+.1f}</div>
  </div>
  <div class="team"><img src="{logo(r['away'])}"><div class="tn">{nombre(r['away'])}</div><div class="loc">Visitante</div></div>
</div>"""


def estilo_tabla(tabla):
    sty = tabla.style.format({"Modelo": "{:.1%}", "Implícita": "{:.1%}", "Cuota": "{:+d}", "Ventaja (pts)": "{:+.1f}"})
    aplicar = getattr(sty, "map", None) or sty.applymap
    sty = aplicar(lambda v: "color:#0f9d58;font-weight:700" if v > 0 else "color:#d93025", subset=["Ventaja (pts)"])
    aplicar = getattr(sty, "map", None) or sty.applymap
    return aplicar(lambda v: "background-color:#d9f5e6;color:#0b6b3a;font-weight:700" if v == "CASHHH" else "",
                   subset=["Señal"])


def serie_balance(df):
    d = df.copy()
    if d.empty:
        return pd.DataFrame()
    d["gan"] = d.apply(ganancia, axis=1)
    d = d[d.resultado.isin(["Ganado", "Perdido", "Push"])]
    if d.empty:
        return pd.DataFrame()
    s = d.groupby("fecha").gan.sum().sort_index()
    out = pd.DataFrame({"Balance diario": s, "Acumulado": s.cumsum()})
    out.index = pd.to_datetime(out.index)
    return out


# ------------------------------------------------------------------------ UI
def ui():
    import streamlit as st

    st.set_page_config(page_title="NBA Analytics", page_icon="🏀", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(hero_html(), unsafe_allow_html=True)
    st.caption("Ganador, spread, totales y props de jugador · el modelo es una segunda opinión, no una garantía")

    try:
        clave = st.secrets.get("password")
    except Exception:
        clave = None
    if clave and not st.session_state.get("acceso"):
        ingreso = st.text_input("Contraseña", type="password")
        if ingreso and ingreso == str(clave):
            st.session_state["acceso"] = True
            st.rerun()
        elif ingreso:
            st.error("Contraseña incorrecta")
        st.stop()

    @st.cache_resource(show_spinner="Cargando modelo y datos…")
    def datos():
        b, d = cargar_modelo()
        return b, d, cargar_jugadores(b)

    if not (m.MODELS / "game.joblib").exists():
        st.error("Todavía no hay modelo entrenado. Corre 'python nba_model.py train' en la terminal.")
        return
    with st.sidebar:
        st.header("Datos")
        if m.NUBE:
            st.caption("Versión en línea: los datos se actualizan desde la PC del dueño.")
        else:
            if st.button("Actualizar partidos recientes (rápido)"):
                st.cache_resource.clear()
                st.rerun()
            st.caption("Descarga los últimos partidos, sin reentrenar. Tarda unos segundos.")
            if st.button("Actualizar datos y reentrenar"):
                with st.spinner("Descargando partidos y entrenando (unos minutos)…"):
                    m.train(m.default_seasons(), 10)
                    st.cache_resource.clear()
                st.success("Listo. Recarga la página.")
            st.caption("Hazlo una vez por semana. Las lesiones no las conoce el modelo: usa las casillas de bajas.")
    b, d, pl = datos()

    @st.cache_data(ttl=300, show_spinner="Trayendo las bajas de ESPN…")
    def bajas_hoy():
        return bajas_espn()
    st.sidebar.caption(f"Datos hasta el {d.GAME_DATE.max():%d/%m/%Y}")
    equipos = sorted(d.TEAM_ABBREVIATION.unique())
    t1, t2, t3 = st.tabs(["Partido", "Props de jugador", "Mis picks"])

    def mostrar(tabla):
        st.dataframe(estilo_tabla(tabla), hide_index=True)

    def registrar(clave, tabla, prefijo):
        if m.NUBE:
            return
        st.markdown("**Registrar un pick**")
        c1, c2, c3 = st.columns([3, 1, 1])
        sel = c1.selectbox("Mercado", tabla["Mercado"].tolist(), key=clave + "_sel")
        monto = c2.number_input("Monto", min_value=0.0, value=100.0, step=10.0, key=clave + "_monto")
        c3.write("")
        if c3.button("Guardar pick", key=clave + "_btn"):
            fila_ = tabla[tabla["Mercado"] == sel].iloc[0]
            agregar_pick(f"{prefijo} {sel}".strip(), int(fila_["Cuota"]), monto)
            st.success("Guardado en 'Mis picks'.")

    # ---------------------------------------------------------------- Partido
    with t1:
        c1, c2 = st.columns(2)
        home = c1.selectbox("Local", equipos, format_func=nombre,
                            index=equipos.index("LAL") if "LAL" in equipos else 0)
        c1.image(logo(home), width=70)
        away = c2.selectbox("Visitante", equipos, format_func=nombre,
                            index=equipos.index("POR") if "POR" in equipos else 1)
        c2.image(logo(away), width=70)
        usar_auto = st.checkbox("Bajas y localía automáticas", value=True,
                                help="Trae las lesiones de ESPN y ajusta la ventaja de local con los partidos recientes")
        th = ta = None
        if usar_auto:
            try:
                todas = bajas_hoy()
                th = estimar_bajas(pl, d, home, todas.get(home, []))
                ta = estimar_bajas(pl, d, away, todas.get(away, []))
            except Exception as e:
                st.warning(f"No se pudieron traer las bajas de ESPN ({type(e).__name__}). Se usa 0; "
                           "puedes ajustarlas a mano abajo.")
        kj_h, kj_a = f"juegan_h_{home}_{away}", f"juegan_a_{home}_{away}"
        inj_h = total_bajas(th, st.session_state.get(kj_h, [])) + st.session_state.get("man_h", 0.0)
        inj_a = total_bajas(ta, st.session_state.get(kj_a, [])) + st.session_state.get("man_a", 0.0)
        loc, loc_info = localia_auto(d, home, away) if usar_auto else (0.0, "")
        peso = st.slider("Peso del modelo frente a la línea de la casa", 0.0, 1.0, 0.3, 0.05)
        with st.expander("Bajas detectadas y localía (se calculan solas)", expanded=True):
            for etiqueta, t_, k in ((nombre(home), th, kj_h), (nombre(away), ta, kj_a)):
                st.markdown(f"**{etiqueta}**")
                if t_ is None or t_.empty:
                    st.caption("Sin bajas reportadas.")
                else:
                    st.dataframe(t_.round(2), hide_index=True)
                    st.multiselect("Suponer que SÍ juegan", t_["Jugador"].tolist(), key=k)
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Bajas {home}", f"{inj_h:+.1f}")
            m2.metric(f"Bajas {away}", f"{inj_a:+.1f}")
            m3.metric("Localía automática", f"{loc:+.1f}")
            st.caption((loc_info + " " if loc_info else "") + "Las bajas son una estimación propia a partir de la "
                       "producción reciente de cada jugador y de su estado en ESPN (negativo = refuerzo).")
            a1, a2 = st.columns(2)
            a1.number_input("Ajuste manual bajas local (opcional)", -15.0, 15.0, 0.0, 0.5, key="man_h")
            a2.number_input("Ajuste manual bajas visitante (opcional)", -15.0, 15.0, 0.0, 0.5, key="man_a")
        st.markdown("**Líneas y momios** (deja en 0 los mercados que no quieras)")
        g1, g2, g3 = st.columns(3)
        with g1:
            ml_h = st.number_input("Momio ganador local", value=0, step=5)
            ml_a = st.number_input("Momio ganador visitante", value=0, step=5)
        with g2:
            spread = st.number_input("Spread del local (ej. -3.5)", value=0.0, step=0.5)
            o_sh = st.number_input("Momio spread local", value=-110, step=1)
            o_sa = st.number_input("Momio spread visitante", value=-110, step=1)
        with g3:
            total = st.number_input("Total del partido", value=0.0, step=0.5)
            o_over = st.number_input("Momio over", value=-110, step=1)
            o_under = st.number_input("Momio under", value=-110, step=1)
        with st.expander("Puntos por equipo (opcional)"):
            e1, e2 = st.columns(2)
            tot_h = e1.number_input("Puntos del local (línea)", value=0.0, step=0.5)
            oh1 = e1.number_input("Momio over local", value=-110, step=1)
            oh2 = e1.number_input("Momio under local", value=-110, step=1)
            tot_a = e2.number_input("Puntos del visitante (línea)", value=0.0, step=0.5)
            oa1 = e2.number_input("Momio over visitante", value=-110, step=1)
            oa2 = e2.number_input("Momio under visitante", value=-110, step=1)
        if st.button("Analizar partido", type="primary"):
            if home == away:
                st.warning("Elige dos equipos distintos.")
            else:
                st.session_state["partido"] = analizar_partido(
                    b, d, home, away,
                    spread=spread if spread != 0 else None, o_sh=o_sh, o_sa=o_sa,
                    total=total if total != 0 else None, o_over=o_over, o_under=o_under,
                    ml_h=ml_h or None, ml_a=ml_a or None,
                    tot_h=tot_h if tot_h != 0 else None, o_hover=oh1, o_hunder=oh2,
                    tot_a=tot_a if tot_a != 0 else None, o_aover=oa1, o_aunder=oa2,
                    inj_h=inj_h, inj_a=inj_a, peso=peso, localia=loc)
        r = st.session_state.get("partido")
        if r:
            st.markdown(tarjeta_partido(r), unsafe_allow_html=True)
            st.caption(f"Modelo puro: margen {r['mar0']:+.1f}, total {r['tot0']:.1f}. "
                       f"Ajustado con bajas, localía y línea: margen {r['mar']:+.1f}, total {r['tot']:.1f}.")
            if not r["tabla"].empty:
                mostrar(r["tabla"])
                registrar("p", r["tabla"], f"{r['home']}-{r['away']}")

    # ------------------------------------------------------------------ Props
    with t2:
        nombres = sorted(pl.PLAYER_NAME.unique())
        c1, c2, c3 = st.columns(3)
        jugador = c1.selectbox("Jugador", nombres)
        stat = c2.selectbox("Estadística", m.PSTATS)
        opp = c3.selectbox("Rival", equipos, key="opp", format_func=nombre)
        c3.image(logo(opp), width=50)
        c4, c5, c6 = st.columns(3)
        linea = c4.number_input("Línea", value=20.5, step=0.5)
        oo = c5.number_input("Momio over", value=-110, step=1, key="po")
        ou_ = c6.number_input("Momio under", value=-110, step=1, key="pu")
        c7, c8, c9, c10 = st.columns(4)
        local = c7.checkbox("Juega de local")
        minutos = c8.number_input("Minutos esperados (0 = tendencia)", 0.0, 48.0, 0.0, 1.0,
                                  help="Mira sus últimos 5 partidos")
        prom = c9.number_input("Promedio de esta temporada (0 = no usar)", 0.0, 60.0, 0.0, 0.1)
        pa_ = c10.slider("Peso de ese promedio", 0.0, 1.0, 0.5, 0.05)
        if st.button("Analizar jugador", type="primary"):
            st.session_state["prop"] = analizar_prop(
                b, d, pl, jugador, stat, linea, oo, ou_, opp, local,
                minutos=minutos or None, promedio=prom or None, peso_actual=pa_)
            st.session_state["prop_nombre"] = jugador
        r = st.session_state.get("prop")
        if r:
            st.subheader(f"{st.session_state['prop_nombre']} ({nombre(r['equipo'])}) · {stat}")
            k1, k2, k3 = st.columns(3)
            k1.metric("Proyección", f"{r['pred']:.1f}")
            k2.metric("Desviación", f"{r['sd']:.1f}")
            k3.metric("Minutos de tendencia", f"{r['min_tend']:.1f}")
            mostrar(r["tabla"])
            registrar("pr", r["tabla"], f"{st.session_state['prop_nombre']} {stat}")

    # ------------------------------------------------------------- Mis picks
    with t3:
        if m.NUBE:
            st.info("El registro de picks solo funciona en la versión instalada en tu PC.")
            return
        df = cargar_picks()
        st.caption("Edita la tabla: elige el resultado de cada pick y se guarda sola. "
                   "Puedes añadir filas con el + de abajo.")
        ed = st.data_editor(
            df, num_rows="dynamic", hide_index=True, key="picks_ed",
            column_config={
                "fecha": st.column_config.DateColumn("Fecha"),
                "pick": st.column_config.TextColumn("Pick"),
                "momio": st.column_config.NumberColumn("Momio", format="%d"),
                "monto": st.column_config.NumberColumn("Monto", format="$%.2f"),
                "resultado": st.column_config.SelectboxColumn("Resultado", options=RESULTADOS),
            })
        if not ed.equals(df):
            guardar_picks(ed)
            df = cargar_picks()
        dia = st.date_input("Día a consultar", value=pd.Timestamp.today().date())
        s = resumen(df, dia)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Balance del día", f"${s['dia_bal']:+,.2f}",
                  "Ganaste" if s["dia_bal"] > 0 else ("Perdiste" if s["dia_bal"] < 0 else None))
        k2.metric("Picks del día", s["dia_n"])
        k3.metric("Apostado (resueltos)", f"${s['dia_apostado']:,.2f}")
        k4.metric("ROI del día", f"{(s['dia_bal'] / s['dia_apostado']) if s['dia_apostado'] else 0:+.1%}")
        st.divider()
        j1, j2, j3, j4 = st.columns(4)
        j1.metric("Balance total", f"${s['tot_bal']:+,.2f}")
        j2.metric("Récord", f"{s['ganados']}-{s['perdidos']}")
        j3.metric("ROI total", f"{(s['tot_bal'] / s['tot_apostado']) if s['tot_apostado'] else 0:+.1%}")
        j4.metric("Dinero en juego", f"${s['pendiente']:,.2f}")
        serie = serie_balance(df)
        if not serie.empty:
            st.subheader("Evolución del balance")
            g1, g2 = st.columns(2)
            g1.caption("Acumulado")
            g1.line_chart(serie["Acumulado"])
            g2.caption("Balance por día")
            g2.bar_chart(serie["Balance diario"])
        if s["ganados"] + s["perdidos"] < 30:
            st.caption("Con menos de ~30 picks resueltos la muestra es muy pequeña para sacar conclusiones.")


if __name__ == "__main__":
    ui()
