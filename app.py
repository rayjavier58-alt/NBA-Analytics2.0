#!/usr/bin/env python3
"""
Interfaz del modelo NBA (Streamlit). Va en la MISMA carpeta que nba_model.py.

    pip install streamlit
    streamlit run app.py

Se abre sola en el navegador (http://localhost:8501).
"""
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


def cargar_jugadores(b):
    return m.add_pra(m.fetch_logs(b["seasons"], "P"))


def analizar_partido(b, d, home, away, *, spread=None, o_sh=-110, o_sa=-110, total=None, o_over=-110,
                     o_under=-110, ml_h=None, ml_a=None, tot_h=None, o_hover=-110, o_hunder=-110,
                     tot_a=None, o_aover=-110, o_aunder=-110, inj_h=0.0, inj_a=0.0, peso=0.3, fecha=None):
    fecha = pd.Timestamp(fecha) if fecha else pd.Timestamp.today().normalize()
    X = m.make_X(m.team_state(d, home, fecha), m.team_state(d, away, fecha))
    mar0, tot0 = float(b["margin"].predict(X)[0]), float(b["total"].predict(X)[0])
    mar, tot = mar0 - inj_h + inj_a, tot0 - 0.5 * (inj_h + inj_a)
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
    return {"home": H, "away": V, "mar0": mar0, "tot0": tot0, "mar": mar, "tot": tot, "ph": ph,
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
.block-container{padding-top:1.4rem;max-width:1200px}
.hero{background:linear-gradient(135deg,#0b1f3a,#17407a);color:#fff;padding:20px 26px;border-radius:14px;margin-bottom:14px}
.hero h1{margin:0;font-size:30px;color:#fff;padding:0}
.hero p{margin:4px 0 0;opacity:.8}
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

HERO = """
<div class="hero"><h1>🏀 Analítica NBA</h1>
<p>Ganador, spread, totales y props de jugador · el modelo es una segunda opinión, no una garantía</p></div>
"""


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

    st.set_page_config(page_title="Analítica NBA", page_icon="🏀", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(HERO, unsafe_allow_html=True)

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
            if st.button("Actualizar datos y reentrenar"):
                with st.spinner("Descargando partidos y entrenando (unos minutos)…"):
                    m.train(m.default_seasons(), 10)
                    st.cache_resource.clear()
                st.success("Listo. Recarga la página.")
            st.caption("Hazlo una vez por semana. Las lesiones no las conoce el modelo: usa las casillas de bajas.")
    b, d, pl = datos()
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
        c1, c2, c3, c4 = st.columns(4)
        home = c1.selectbox("Local", equipos, format_func=nombre,
                            index=equipos.index("LAL") if "LAL" in equipos else 0)
        c1.image(logo(home), width=70)
        away = c2.selectbox("Visitante", equipos, format_func=nombre,
                            index=equipos.index("POR") if "POR" in equipos else 1)
        c2.image(logo(away), width=70)
        inj_h = c3.number_input("Bajas local (net rating)", 0.0, 15.0, 0.0, 0.5,
                                help="Estrella 3-5, rotación 1-2")
        inj_a = c4.number_input("Bajas visitante (net rating)", 0.0, 15.0, 0.0, 0.5)
        peso = st.slider("Peso del modelo frente a la línea de la casa", 0.0, 1.0, 0.3, 0.05)
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
                    inj_h=inj_h, inj_a=inj_a, peso=peso)
        r = st.session_state.get("partido")
        if r:
            st.markdown(tarjeta_partido(r), unsafe_allow_html=True)
            st.caption(f"Modelo puro: margen {r['mar0']:+.1f}, total {r['tot0']:.1f}. "
                       f"Ajustado con bajas y línea: margen {r['mar']:+.1f}, total {r['tot']:.1f}.")
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
