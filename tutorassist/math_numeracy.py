import streamlit as st
from numerace import numerace_app

st.set_page_config(page_title="NumeRace", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.35rem;
    }

    .nr-topbar{
      padding: 6px 4px 2px 4px;
      background: transparent;
      border: none;
      border-radius: 0;
      box-shadow: none;
    }

    .nr-title {
      font-size: 34px;
      font-weight: 800;
      letter-spacing: 0.5px;
      text-align: center;
      margin-top: -2px;
    }

    .nr-sub {
      font-size: 14px;
      opacity: 0.82;
      margin-top: -6px;
      text-align: center;
    }

    .nr-card{
        border: none;
        border-radius: 0;
        padding: 2px 0 0 0;
        background: transparent;
        box-shadow: none;
    }

    .nr-qwrap {
        max-width: 760px;
        margin-left:auto;
        margin-right:auto;
    }

    .nr-prompt{
        text-align:center;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .nr-muted {
        opacity: 0.75;
    }

    .nr-controls{
      display:flex;
      flex-direction:column;
      gap:0.15rem;
      padding-top: 0;
    }

    .nr-controls-label{
      font-size: 13px;
      font-weight: 700;
      opacity: 0.78;
      margin-bottom: -0.05rem;
    }

    .nr-sep{
      height: 2px;
      width: 100%;
      margin: 12px 0 16px 0;
      background: linear-gradient(
        to right,
        rgba(15,23,42,0.00),
        rgba(15,23,42,0.22),
        rgba(15,23,42,0.28),
        rgba(15,23,42,0.22),
        rgba(15,23,42,0.00)
      );
    }

    .nr-track-flags{
      display:flex;
      justify-content: space-between;
      align-items:center;
      padding: 0 2px 6px 2px;
    }

    .nr-flag{
      font-size: 20px;
      line-height: 1;
      opacity: 0.85;
    }

    .nr-track-stage{
      position: relative;
      overflow: visible;
    }

    .nr-track-line{
      height: 10px;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.12);
      position: relative;
      overflow: hidden;
    }

    .nr-track-line::after{
      content: "";
      position: absolute;
      inset: 0;
      border-radius: 999px;
      pointer-events: none;
      background-image: repeating-linear-gradient(
        to right,
        transparent 0,
        transparent calc(12.5% - 1px),
        rgba(255,255,255,0.70) calc(12.5% - 1px),
        rgba(255,255,255,0.70) calc(12.5% + 1px)
      );
    }

    .nr-track-fill{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg,#3b82f6,#2563eb);
    }

    .nr-racer{
      position:absolute;
      top:-16px;
      transform: translateX(-50%) scaleX(-1);
      font-size: 36px;
      transition: left 0.45s ease;
      will-change: left, transform;
      z-index: 5;
      animation: nr-racer-idle 1.35s ease-in-out infinite;
    }

    .nr-racer-finish{
      animation:
        nr-racer-idle 1.35s ease-in-out infinite,
        nr-finish-burst 0.75s ease-out 1;
    }

    .nr-smoke{
      position:absolute;
      top:-6px;
      transform: translateX(-50%);
      font-size: 22px;
      opacity: 0.82;
      z-index: 4;
      pointer-events: none;
      animation: nr-smoke-puff 0.9s ease-in-out infinite;
    }

    @keyframes nr-racer-idle{
      0%   { transform: translateX(-50%) scaleX(-1) translateY(0px); }
      50%  { transform: translateX(-50%) scaleX(-1) translateY(-2px); }
      100% { transform: translateX(-50%) scaleX(-1) translateY(0px); }
    }

    @keyframes nr-smoke-puff{
      0%   { transform: translateX(-50%) scale(0.92); opacity: 0.45; }
      50%  { transform: translateX(-50%) translateX(-2px) scale(1.06); opacity: 0.82; }
      100% { transform: translateX(-50%) translateX(-4px) scale(1.18); opacity: 0.20; }
    }

    @keyframes nr-finish-burst{
      0%   { transform: translateX(-50%) scaleX(-1) translateY(0px) scale(1); }
      35%  { transform: translateX(-50%) scaleX(-1) translateY(-5px) scale(1.22) rotate(5deg); }
      65%  { transform: translateX(-50%) scaleX(-1) translateY(-2px) scale(1.12) rotate(-4deg); }
      100% { transform: translateX(-50%) scaleX(-1) translateY(0px) scale(1.0); }
    }

    .nr-score-pop{
      text-align:center;
      margin-top:0.35rem;
      margin-bottom:1.0rem;
      padding:0.85rem 0.8rem 0.95rem 0.8rem;
      border-radius:18px;
      background:rgba(255,255,255,0.24);
      border:1px solid rgba(15,23,42,0.10);
      animation: nr-score-pop-in 0.42s ease-out 1;
    }

    .nr-score-kicker{
      font-size:1.0rem;
      opacity:0.72;
      margin-bottom:0.15rem;
    }

    .nr-score-value{
      font-size:3.15rem;
      font-weight:800;
      line-height:1.0;
      letter-spacing:0.5px;
    }

    @keyframes nr-score-pop-in{
      0%   { transform: translateY(8px) scale(0.94); opacity: 0; }
      65%  { transform: translateY(-2px) scale(1.03); opacity: 1; }
      100% { transform: translateY(0px) scale(1.0); opacity: 1; }
    }

    .nr-fb {
      margin-top: 10px;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid rgba(15, 23, 42, 0.12);
      background: rgba(255,255,255,0.55);
      font-size: 18px;
      line-height: 1.25;
    }

    .nr-fb small {
      font-size: 13px;
      opacity: 0.8;
    }

    .nr-fb .nr-fb-val {
      font-size: 22px;
      font-weight: 800;
    }

    .nr-fb-ok  {
      border-color: rgba(34,197,94,0.45);
      background: rgba(34,197,94,0.10);
    }

    .nr-fb-bad {
      border-color: rgba(239,68,68,0.45);
      background: rgba(239,68,68,0.08);
    }

    .nr-fb-neu {
      border-color: rgba(148,163,184,0.35);
      background: rgba(148,163,184,0.10);
    }

    div[data-testid="stButton"] > button {
      border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
numerace_app()
