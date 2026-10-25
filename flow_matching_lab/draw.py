"""Editable noise canvas with real CPU Flow Matching inference."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import time
from urllib.parse import parse_qs, urlsplit

import torch

from .core import VelocityMLP, integrate
from .image_model import make_image_model, sample_image


PAGE = r"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>纹路 → 数字 · Flow Matching</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101418;color:#e9eff3;font:16px system-ui,sans-serif}
main{max-width:1000px;margin:auto;padding:32px}h1{font-size:28px;margin:0 0 12px}
p{color:#aebdc8;line-height:1.7} .panels{display:flex;gap:24px;flex-wrap:wrap}
section{background:#1c242c;padding:24px;border:1px solid #34424c;border-radius:16px;flex:1;min-width:280px}
h2{font-size:18px;margin:0 0 16px}canvas{width:100%;max-width:350px;aspect-ratio:1;image-rendering:pixelated;background:#000;border:1px solid #536371;touch-action:none}
.controls{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0;align-items:center}
label{display:flex;gap:8px;align-items:center}button,select,input{font:inherit}
button,select{background:#273642;color:#fff;border:1px solid #536371;padding:8px 14px;border-radius:8px;cursor:pointer}
button:hover{background:#365466}input[type=range]{width:120px;accent-color:#63e4b5}
.status{min-height:28px;color:#63e4b5}.caption{font-size:14px;color:#aebdc8;margin-top:12px}
.active{outline:2px solid #63e4b5}small{color:#aebdc8}a{color:#63e4b5}
</style>
<main><h1>改一笔纹路，看数字怎么变</h1>
<p>在左边按住拖动，修改模型的初始噪声。右边由真实模型重新生成；固定数字类别时，可以观察笔画、粗细与倾斜的变化。</p>
<div class="controls">
<label>目标数字 <select id="digit"></select></label>
<label>步数 <select id="steps"><option>1</option><option>8</option><option selected>32</option><option>64</option></select></label>
<label>CFG <input id="cfg" type="range" min="0" max="3" step=".1" value="1"><output id="cfgv">1</output></label>
<button id="random">换一张噪声</button><button id="reset">恢复本张噪声</button>
</div>
<div class="panels"><section><h2>输入 · 可编辑噪声</h2>
<canvas id="input" width="280" height="280" aria-label="在这里绘制噪声纹路"></canvas>
<div class="controls"><button id="light" class="active">亮笔</button><button id="dark">暗笔</button>
<label>笔刷 <input id="radius" type="range" min="1" max="3" step=".5" value="1"></label>
<label>力度 <input id="strength" type="range" min=".1" max="1" step=".1" value=".5"></label></div>
<small>亮笔增加噪声数值，暗笔减小数值。修改积累在原始噪声上。</small>
</section><section><h2>输出 · 模型生成</h2><canvas id="result" width="280" height="280" aria-label="模型生成数字"></canvas>
<div class="caption" id="difference">等待第一次生成</div>
<div class="controls"><button id="save">保存当前输入和输出</button></div></section></div>
<p class="status" id="status">正在加载模型…</p>
<p>这里编辑的是噪声，不是数字草图。模型没有接受过“照着草图画”的训练，因此局部改动可能带来全局变化。一步生成往往接近类别原型；建议用 32 步、CFG 1 观察输入的影响。</p>
</main>
<script>
const $=id=>document.getElementById(id), n=__SIDE__;
let x=[],original=[],previous=null,output=null,sign=1,drawing=false,last=null;
let revision=0,pending=false,busy=false,timer=null;
for(let i=0;i<10;i++){$('digit').add(new Option(i,i))}$('digit').value='3';
function paint(id,values){const c=$(id),ctx=c.getContext('2d'),cell=c.width/n;
 values.forEach((v,i)=>{let g=Math.round((Math.max(-2,Math.min(2,v))/2+1)*127.5);
 if(id==='result')g=Math.round((Math.max(-1,Math.min(1,v))+1)*127.5);
 ctx.fillStyle=`rgb(${g},${g},${g})`;ctx.fillRect(i%n*cell,Math.floor(i/n)*cell,cell,cell)})}
function normal(){let u=1-Math.random(),v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
function schedule(){revision++;pending=true;clearTimeout(timer);timer=setTimeout(run,120)}
async function run(){if(busy||!pending)return;busy=true;pending=false;
 const current=revision,settings={noise:x.slice(),digit:+$('digit').value,steps:+$('steps').value,cfg:+$('cfg').value};
 $('status').textContent='正在重新生成…';
 try{const response=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(settings)});
 const data=await response.json();if(!response.ok)throw Error(data.error||'生成失败');
 if(current===revision){output=data.samples;paint('result',output);
 let delta=previous?Math.sqrt(output.reduce((s,v,i)=>s+(v-previous[i])**2,0)/output.length):0;
 $('difference').textContent=previous?`与上次输出的像素 RMS 差异：${delta.toFixed(4)}`:'第一张输出已生成';
 previous=output.slice();$('status').textContent=`已更新 · ${Math.round(data.seconds*1000)} ms · ${settings.steps} 步`;
 }}catch(e){$('status').textContent='生成失败：'+e.message}finally{busy=false;if(pending)run()}}
function fresh(){x=Array.from({length:n*n},normal);original=x.slice();paint('input',x);schedule()}
$('random').onclick=fresh;$('reset').onclick=()=>{x=original.slice();paint('input',x);schedule()};
for(const id of ['digit','steps','cfg'])$(id).addEventListener('input',()=>{$('cfgv').value=$('cfg').value;schedule()});
for(const [id,s] of [['light',1],['dark',-1]])$(id).onclick=()=>{sign=s;$('light').classList.toggle('active',s===1);$('dark').classList.toggle('active',s===-1)};
function point(e){let r=$('input').getBoundingClientRect();return [(e.clientX-r.left)/r.width*n,(e.clientY-r.top)/r.height*n]}
function stroke(p){let radius=+$('radius').value,amount=+$('strength').value*.6;
 for(let iy=0;iy<n;iy++)for(let ix=0;ix<n;ix++){let d=Math.hypot(ix+.5-p[0],iy+.5-p[1]);
 if(d<radius)x[iy*n+ix]=Math.max(-4,Math.min(4,x[iy*n+ix]+sign*amount*(1-d/radius)))}
}
$('input').addEventListener('pointerdown',e=>{drawing=true;last=point(e);$('input').setPointerCapture(e.pointerId);stroke(last);paint('input',x);schedule()});
$('input').addEventListener('pointermove',e=>{if(!drawing)return;let p=point(e),d=Math.hypot(p[0]-last[0],p[1]-last[1]),k=Math.max(1,Math.ceil(d*3));
 for(let i=1;i<=k;i++)stroke([last[0]+(p[0]-last[0])*i/k,last[1]+(p[1]-last[1])*i/k]);
 last=p;paint('input',x);schedule()});
for(const event of ['pointerup','pointercancel'])$('input').addEventListener(event,()=>{drawing=false;last=null});
$('save').onclick=async()=>{if(busy||pending||!output){$('status').textContent='请等当前生成完成再保存';return}
 try{const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({noise:x,digit:+$('digit').value,steps:+$('steps').value,cfg:+$('cfg').value})});
 const data=await r.json();if(!r.ok)throw Error(data.error);$('status').textContent='已保存：'+data.path}catch(e){$('status').textContent=e.message}};
fresh();
</script></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    improved_run = Path("flow_matching_lab/improved/unet_seed2026")
    default_run = improved_run if (improved_run / "checkpoint.pt").exists() else Path("flow_matching_lab/results/seed2026")
    parser.add_argument("--run", type=Path, default=default_run)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7866)
    parser.add_argument("--output", type=Path, default=Path("flow_matching_lab/draw_outputs"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(args.run / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = saved["config"]
    side = config["image_size"]
    image_conditioned = "architecture" in config
    model = (make_image_model(config) if image_conditioned
             else VelocityMLP(side ** 2, config["width"], config["depth"])).eval()
    model.load_state_dict(saved["ema"])
    page = ((Path(__file__).parent / "editor.html").read_text() if image_conditioned else PAGE)

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, content, mime="application/json"):
            body = content.encode()
            self.send_response(code)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/":
                self.send(200, page.replace("__SIDE__", str(side)), "text/html")
            elif url.path == "/example" and image_conditioned:
                examples_path = Path(__file__).parent / "improved" / "examples.json"
                try:
                    digit = int(parse_qs(url.query).get("digit", ["3"])[0])
                    if not 0 <= digit <= 9:
                        raise ValueError("Digit must be 0–9")
                    examples = json.loads(examples_path.read_text())
                    self.send(200, json.dumps({"image": examples["images"][str(digit)]}))
                except (ValueError, FileNotFoundError, KeyError) as exc:
                    self.send(400, json.dumps({"error": f"Example unavailable: {exc}"}))
            else:
                self.send(404, json.dumps({"error": "Not found"}))

        def do_POST(self):
            if self.path not in {"/generate", "/save"}:
                self.send(404, json.dumps({"error": "Not found"}))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 100000:
                    raise ValueError("Invalid request length")
                data = json.loads(self.rfile.read(length))
                noise = data["noise"]
                digit, steps, cfg = data["digit"], data["steps"], data["cfg"]
                if (type(digit) is not int or not 0 <= digit <= 9
                        or type(steps) is not int or not 1 <= steps <= 128
                        or not isinstance(cfg, (int, float)) or not math.isfinite(cfg) or not 0 <= cfg <= 3
                        or not isinstance(noise, list) or len(noise) != side ** 2
                        or any(not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 100 for v in noise)):
                    raise ValueError("Invalid noise or generation settings")
                x0 = torch.tensor([noise], dtype=torch.float32)
                known_values, mask_values = data.get("known"), data.get("mask")
                known, mask = torch.zeros_like(x0), torch.zeros_like(x0)
                if image_conditioned and known_values is not None:
                    if (not isinstance(known_values, list) or len(known_values) != side ** 2
                            or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                                   or not -1 <= v <= 1 for v in known_values)
                            or not isinstance(mask_values, list) or len(mask_values) != side ** 2
                            or any(v not in (0, 1) for v in mask_values)):
                        raise ValueError("Invalid observation pixels or binary mask")
                    known = torch.tensor([known_values], dtype=torch.float32)
                    mask = torch.tensor([mask_values], dtype=torch.float32)
                started = time.perf_counter()
                if image_conditioned:
                    solved = sample_image(model, x0.reshape(1, 1, side, side), torch.tensor([digit]),
                                          steps, cfg, known.reshape(1, 1, side, side),
                                          mask.reshape(1, 1, side, side))
                    samples = solved.samples.flatten(1)
                else:
                    samples = integrate(model, x0, torch.tensor([digit]), steps=steps, guidance=cfg).samples
                result = {"samples": samples[0].tolist(), "seconds": time.perf_counter() - started,
                          "known_max_error": float(((samples - known).abs() * mask).max())}
                if self.path == "/save":
                    from PIL import Image
                    folder = args.output / str(time.time_ns())
                    folder.mkdir(parents=True)
                    display_input = known * mask if image_conditioned and mask.any() else x0
                    for name, values in (("input", display_input), ("output", samples), ("mask", mask * 2 - 1)):
                        pixels = ((values.reshape(side, side).clamp(-1, 1) + 1) * 127.5).byte().numpy()
                        Image.fromarray(pixels).resize((280, 280), Image.Resampling.NEAREST).save(folder / f"{name}.png")
                    (folder / "settings.json").write_text(json.dumps(data | result, indent=2) + "\n")
                    result["path"] = str(folder.resolve())
                self.send(200, json.dumps(result))
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                self.send(400, json.dumps({"error": str(exc)}))

    print(f"Open http://{args.host}:{args.port} — edit the left canvas to change generated digits.", flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
