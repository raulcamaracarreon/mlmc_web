from __future__ import annotations
import base64
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, Optional


def fig_to_base64_png(fig) -> str:
    """Convierte una figura de Matplotlib a PNG base64."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _fmt_metrics_ul(metrics: Dict[str, Any]) -> str:
    if not metrics:
        return "<p><em>No hay métricas.</em></p>"
    lis = []
    for k, v in metrics.items():
        if isinstance(v, float):
            lis.append(f"<li><b>{k}</b>: {v:.4f}</li>")
        else:
            lis.append(f"<li><b>{k}</b>: {v}</li>")
    return "<ul>" + "\n".join(lis) + "</ul>"


def render_html_report(*,
                       title: str,
                       task: str,
                       dataset_name: Optional[str],
                       features: Optional[list[str]],
                       target: Optional[str],
                       metrics: Dict[str, Any],
                       extra_text: Optional[str],
                       chart_data_url: Optional[str]) -> str:
    """Devuelve HTML autónomo con estilos embebidos + gráfico (base64) y métricas."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    ds = dataset_name or "(no especificado)"
    feats = ", ".join(features or [])
    tgt = target or "(no aplica)"

    extra_block = f"<pre style='white-space:pre-wrap'>{extra_text}</pre>" if extra_text else ""
    chart_block = f"<img src='{chart_data_url}' alt='chart' style='max-width:100%; height:auto; border:1px solid #e5e7eb; border-radius:12px;'/>" if chart_data_url else "<p><em>Sin gráfico</em></p>"

    METRICS_HTML = _fmt_metrics_ul(metrics)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
  :root {{
    --bg:#0f172a; --card:#111827; --ink:#e5e7eb; --mut:#9ca3af; --accent:#60a5fa; --bd:#1f2937;
  }}
  body {{ margin:0; background:linear-gradient(180deg,#0b1220,#0f172a); color:var(--ink); font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, Arial; }}
  .wrap {{ max-width: 980px; margin: 24px auto; padding: 16px; }}
  .card {{ background: linear-gradient(180deg,#0f172a,#0b1220); border:1px solid var(--bd); border-radius: 16px; padding: 18px; }}
  h1 {{ margin: 0 0 6px 0; font-size: 24px; }}
  .mut {{ color: var(--mut); }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
  .meta b {{ color: var(--ink); }}
  .badge {{ display:inline-block; padding:4px 8px; border-radius:999px; border:1px solid var(--bd); color:var(--mut); }}
  ul {{ margin:8px 0 0 18px; }}
  pre {{ background:#0b1020; border:1px solid var(--bd); padding:12px; border-radius:12px; color:#d1d5db; overflow:auto; }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>{title}</h1>
      <div class="mut">Generado: {now}</div>
    </div>

    <div class="grid" style="margin-top:14px">
      <div class="card meta">
        <div><span class="badge">Tarea</span> <b>{task}</b></div>
        <div style="margin-top:6px"><span class="badge">Dataset</span> <b>{ds}</b></div>
        <div style="margin-top:6px"><span class="badge">Target</span> <b>{tgt}</b></div>
        <div style="margin-top:6px"><span class="badge">Features</span> <span class="mut">{feats}</span></div>
      </div>

      <div class="card">
        <h2 style="margin:0 0 8px 0; font-size:18px">Métricas</h2>
        {METRICS_HTML}
      </div>

      <div class="card">
        <h2 style="margin:0 0 8px 0; font-size:18px">Gráfico</h2>
        {chart_block}
      </div>

      {f'<div class="card"><h2 style="margin:0 0 8px 0; font-size:18px">Detalle</h2>{extra_block}</div>' if extra_text else ''}
    </div>
  </div>
</body>
</html>"""
