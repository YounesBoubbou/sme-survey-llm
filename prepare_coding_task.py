import csv
import json
import os
import random
from collections import Counter

random.seed(42)

PREDICTIONS_FILE = "data/llm_predictions.csv"
CODING_DIR = "coding_task"
N_RESPONSES = 150

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SME AI Survey — Coding Task</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f0f2f5;color:#1a1a2e;min-height:100vh}
.topbar{background:#1a237e;color:#fff;padding:14px 24px;display:flex;justify-content:space-between;align-items:center}
.topbar h1{font-size:16px;font-weight:600;letter-spacing:.2px}
.topbar span{font-size:13px;opacity:.8}
.pbar-wrap{background:#283593;height:5px}
.pbar{background:#64b5f6;height:100%;transition:width .3s ease}
.wrap{max-width:780px;margin:0 auto;padding:24px 16px}

/* setup */
.card{background:#fff;border-radius:10px;padding:36px;box-shadow:0 2px 12px rgba(0,0,0,.08)}
.card h2{font-size:20px;color:#1a237e;margin-bottom:10px}
.card p{color:#555;line-height:1.65;margin-bottom:14px;font-size:14px}
.card input{width:100%;padding:10px 14px;border:1.5px solid #ccc;border-radius:7px;font-size:15px;margin-bottom:16px;outline:none}
.card input:focus{border-color:#3f51b5}

/* instructions */
details.hints{background:#e8eaf6;border-left:4px solid #3f51b5;border-radius:6px;padding:12px 16px;margin-bottom:18px;font-size:13px}
details.hints summary{cursor:pointer;font-weight:600;color:#1a237e;user-select:none}
.def-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px}
.def-col h4{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:#3f51b5;margin-bottom:7px}
.def-col p{margin:3px 0;color:#444;line-height:1.5}
.def-col b{font-weight:600;color:#222}
.def-full{grid-column:1/-1}

/* response */
.resp-card{background:#fff;border-radius:10px;padding:22px 24px;box-shadow:0 2px 12px rgba(0,0,0,.08);margin-bottom:18px}
.resp-lbl{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#9e9e9e;margin-bottom:10px}
.resp-text{font-size:15.5px;line-height:1.75;color:#111;border-left:3px solid #3f51b5;padding-left:16px}

/* label panels */
.lbls-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.lbl-panel{background:#fff;border-radius:10px;padding:18px 20px;box-shadow:0 2px 12px rgba(0,0,0,.08);border:2px solid transparent;transition:border-color .15s}
.lbl-panel.done{border-color:#c5cae9}
.lbl-panel.err{border-color:#ef5350}
.lbl-panel.full{grid-column:1/-1}
.lbl-panel h3{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:#9e9e9e;margin-bottom:12px}
.opt{display:flex;align-items:flex-start;gap:10px;padding:7px 9px;border-radius:6px;cursor:pointer;transition:background .12s;margin-bottom:3px}
.opt:hover{background:#f0f4ff}
.opt input{margin-top:2px;width:15px;height:15px;cursor:pointer;accent-color:#3f51b5;flex-shrink:0}
.opt label{cursor:pointer;font-size:14px;font-weight:500;line-height:1.4}
.opt label small{display:block;font-size:12px;color:#9e9e9e;font-weight:400;margin-top:1px}
.barrier-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}

/* nav */
.nav{display:flex;justify-content:space-between;align-items:center;margin-top:6px}
.btn{padding:10px 26px;border-radius:7px;font-size:14px;font-weight:600;cursor:pointer;border:none;transition:all .15s}
.btn-p{background:#3f51b5;color:#fff}.btn-p:hover{background:#303f9f}
.btn-s{background:#eeeeee;color:#444}.btn-s:hover{background:#e0e0e0}
.btn:disabled{opacity:.35;cursor:not-allowed}
.errmsg{color:#ef5350;font-size:13px;text-align:center;padding:8px;display:none}

/* complete */
.complete{text-align:center;padding:52px 36px}
.complete h2{font-size:22px;color:#1a237e;margin-bottom:10px}
.complete p{color:#555;margin-bottom:20px;line-height:1.65;font-size:14px}
.complete .stat-box{background:#f5f5f5;border-radius:8px;padding:14px;margin-bottom:24px;font-size:14px;color:#444}
.btn-dl{background:#1b5e20;color:#fff;padding:14px 36px;font-size:15px;border-radius:8px;margin:0 auto;display:block}
.btn-dl:hover{background:#2e7d32}
</style>
</head>
<body>
<div class="topbar">
  <h1>SME AI Adoption Survey — Coding Task</h1>
  <span id="ptxt"></span>
</div>
<div class="pbar-wrap"><div class="pbar" id="pbar" style="width:0"></div></div>
<div class="wrap">

<!-- SETUP -->
<div id="s-setup" class="card">
  <h2>Welcome</h2>
  <p>You are being asked to independently label <strong>__N__ survey responses</strong> from SME owners about AI adoption. This should take approximately <strong>20–30 minutes</strong>.</p>
  <p>For each response, assign three labels: <strong>sentiment</strong>, <strong>adoption stage</strong>, and <strong>main barrier</strong>. Full definitions are shown on every screen. Work independently — do not discuss labels with others until you have finished.</p>
  <p>Your progress is automatically saved in this browser. You can close the tab and resume later on the same device.</p>
  <input type="text" id="coder-id" placeholder="Enter your name or initials" autocomplete="off">
  <button class="btn btn-p" onclick="startTask()" style="width:100%">Start Coding →</button>
</div>

<!-- CODING -->
<div id="s-coding" style="display:none">

  <details class="hints" open>
    <summary>Label Definitions — click to collapse</summary>
    <div class="def-grid">
      <div class="def-col">
        <h4>Sentiment</h4>
        <p><b>Positive</b> — optimistic, satisfied, or enthusiastic about AI</p>
        <p><b>Neutral</b> — mixed feelings, uncertain, or cautious</p>
        <p><b>Negative</b> — frustrated, dismissive, or opposed to AI</p>
      </div>
      <div class="def-col">
        <h4>Adoption Stage</h4>
        <p><b>Exploring</b> — researching or considering AI, not yet trialling</p>
        <p><b>Piloting</b> — actively testing or trialling AI tools</p>
        <p><b>Scaling</b> — AI is deployed and being expanded</p>
        <p><b>Not interested</b> — no intention to adopt AI</p>
      </div>
      <div class="def-col def-full">
        <h4>Main Barrier</h4>
        <p><b>Cost</b> — financial constraints or pricing concerns &nbsp;·&nbsp; <b>Skills</b> — lack of expertise or training needs &nbsp;·&nbsp; <b>Trust</b> — concerns about reliability, privacy, or accuracy &nbsp;·&nbsp; <b>None</b> — no significant barrier mentioned</p>
      </div>
    </div>
  </details>

  <div class="resp-card">
    <div class="resp-lbl">Survey Response</div>
    <div class="resp-text" id="resp-text"></div>
  </div>

  <div class="lbls-grid">
    <div class="lbl-panel" id="p-sentiment">
      <h3>Sentiment</h3>
      <div class="opt"><input type="radio" name="sentiment" id="s-pos" value="positive"><label for="s-pos">Positive<small>optimistic / satisfied</small></label></div>
      <div class="opt"><input type="radio" name="sentiment" id="s-neu" value="neutral"><label for="s-neu">Neutral<small>mixed / uncertain</small></label></div>
      <div class="opt"><input type="radio" name="sentiment" id="s-neg" value="negative"><label for="s-neg">Negative<small>frustrated / opposed</small></label></div>
    </div>
    <div class="lbl-panel" id="p-stage">
      <h3>Adoption Stage</h3>
      <div class="opt"><input type="radio" name="stage" id="st-exp" value="exploring"><label for="st-exp">Exploring<small>researching, not yet trialling</small></label></div>
      <div class="opt"><input type="radio" name="stage" id="st-pil" value="piloting"><label for="st-pil">Piloting<small>actively testing tools</small></label></div>
      <div class="opt"><input type="radio" name="stage" id="st-scl" value="scaling"><label for="st-scl">Scaling<small>deployed and expanding</small></label></div>
      <div class="opt"><input type="radio" name="stage" id="st-noi" value="not_interested"><label for="st-noi">Not interested<small>no intention to adopt</small></label></div>
    </div>
    <div class="lbl-panel full" id="p-barrier">
      <h3>Main Barrier</h3>
      <div class="barrier-grid">
        <div class="opt"><input type="radio" name="barrier" id="b-cost" value="cost"><label for="b-cost">Cost<small>financial</small></label></div>
        <div class="opt"><input type="radio" name="barrier" id="b-skl" value="skills"><label for="b-skl">Skills<small>expertise gap</small></label></div>
        <div class="opt"><input type="radio" name="barrier" id="b-tru" value="trust"><label for="b-tru">Trust<small>reliability / privacy</small></label></div>
        <div class="opt"><input type="radio" name="barrier" id="b-non" value="none"><label for="b-non">None<small>no barrier</small></label></div>
      </div>
    </div>
  </div>

  <div class="errmsg" id="errmsg">Please select a value for all three labels before continuing.</div>
  <div class="nav">
    <button class="btn btn-s" id="btn-back" onclick="go(-1)" disabled>← Back</button>
    <button class="btn btn-p" onclick="go(1)">Save &amp; Continue →</button>
  </div>
</div>

<!-- COMPLETE -->
<div id="s-complete" class="card complete" style="display:none">
  <h2>Coding Complete</h2>
  <p>Thank you. Please download your responses and send the CSV file back to the researcher.</p>
  <div class="stat-box" id="stat-box"></div>
  <button class="btn btn-dl" onclick="exportCSV()">⬇ Download CSV</button>
</div>

</div><!-- /wrap -->
<script>
const R=__RESPONSES_JSON__;
const KEY='sme-irr-v1';
let st={id:'',lbl:{},idx:0};

function save(){localStorage.setItem(KEY,JSON.stringify(st));}
function load(){try{const s=localStorage.getItem(KEY);if(s)st=JSON.parse(s);}catch(e){}}

function startTask(){
  const nm=document.getElementById('coder-id').value.trim();
  if(!nm){alert('Please enter your name or initials.');return;}
  load();
  if(st.id===nm&&Object.keys(st.lbl).length>0){
    if(!confirm(`Welcome back, ${nm}! Resume from response ${st.idx+1} of ${R.length}?`))
      st={id:nm,lbl:{},idx:0};
  } else {
    st={id:nm,lbl:{},idx:0};
  }
  save();
  document.getElementById('s-setup').style.display='none';
  document.getElementById('s-coding').style.display='block';
  render();
}

function render(){
  const i=st.idx;
  if(i>=R.length){done();return;}
  const r=R[i];
  document.getElementById('resp-text').textContent=r.response;
  ['sentiment','stage','barrier'].forEach(n=>
    document.querySelectorAll(`input[name="${n}"]`).forEach(el=>el.checked=false));
  const sv=st.lbl[r.row_id];
  if(sv){
    const s=document.getElementById('s-'+{positive:'pos',neutral:'neu',negative:'neg'}[sv.sentiment]);
    const st2=document.getElementById('st-'+{exploring:'exp',piloting:'pil',scaling:'scl',not_interested:'noi'}[sv.adoption_stage]);
    const b=document.getElementById('b-'+{cost:'cost',skills:'skl',trust:'tru',none:'non'}[sv.main_barrier]);
    if(s)s.checked=true; if(st2)st2.checked=true; if(b)b.checked=true;
  }
  ['p-sentiment','p-stage','p-barrier'].forEach(id=>{
    const el=document.getElementById(id);
    el.classList.remove('err');
    el.classList.toggle('done',!!sv);
  });
  document.getElementById('errmsg').style.display='none';
  const pct=Math.round(i/R.length*100);
  document.getElementById('ptxt').textContent=`Response ${i+1} of ${R.length}`;
  document.getElementById('pbar').style.width=pct+'%';
  document.getElementById('btn-back').disabled=i===0;
}

function go(dir){
  if(dir===1){
    const sentiment=document.querySelector('input[name="sentiment"]:checked')?.value;
    const stage=document.querySelector('input[name="stage"]:checked')?.value;
    const barrier=document.querySelector('input[name="barrier"]:checked')?.value;
    let err=false;
    if(!sentiment){document.getElementById('p-sentiment').classList.add('err');err=true;}
    if(!stage){document.getElementById('p-stage').classList.add('err');err=true;}
    if(!barrier){document.getElementById('p-barrier').classList.add('err');err=true;}
    if(err){document.getElementById('errmsg').style.display='block';return;}
    const r=R[st.idx];
    st.lbl[r.row_id]={sentiment,adoption_stage:stage,main_barrier:barrier};
    st.idx++;save();
    if(st.idx>=R.length){done();return;}
  } else {
    if(st.idx>0){st.idx--;save();}
  }
  render();
}

function done(){
  document.getElementById('s-coding').style.display='none';
  document.getElementById('s-complete').style.display='block';
  document.getElementById('pbar').style.width='100%';
  document.getElementById('ptxt').textContent=`Complete — ${R.length} responses`;
  const n=Object.keys(st.lbl).length;
  document.getElementById('stat-box').textContent=
    `Coder: ${st.id}  |  Responses coded: ${n} / ${R.length}`;
}

function exportCSV(){
  const lines=['coder_id,row_id,response,sentiment,adoption_stage,main_barrier'];
  R.forEach(r=>{
    const l=st.lbl[r.row_id];
    if(!l)return;
    lines.push([st.id,r.row_id,'"'+r.response.replace(/"/g,'""')+'"',
      l.sentiment,l.adoption_stage,l.main_barrier].join(','));
  });
  const blob=new Blob([lines.join('\n')],{type:'text/csv'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download=`coder_${st.id.replace(/\s+/g,'_').toLowerCase()}.csv`;
  a.click();
}

window.addEventListener('load',()=>{
  load();
  if(st.id)document.getElementById('coder-id').value=st.id;
});
</script>
</body>
</html>"""


def load_predictions():
    with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_errors(row):
    return sum([
        row["true_sentiment"] != row["pred_sentiment"],
        row["true_adoption_stage"] != row["pred_adoption_stage"],
        row["true_main_barrier"] != row["pred_main_barrier"],
    ])


def select_responses(rows, n=150):
    error_rows = [r for r in rows if count_errors(r) > 0]
    correct_rows = [r for r in rows if count_errors(r) == 0]
    random.shuffle(error_rows)
    random.shuffle(correct_rows)
    selected = error_rows[:n] + correct_rows[:max(0, n - len(error_rows))]
    random.shuffle(selected)
    return selected[:n]


def main():
    os.makedirs(CODING_DIR, exist_ok=True)
    rows = load_predictions()
    selected = select_responses(rows, N_RESPONSES)

    # Save ground-truth copy for analysis (NOT shared with coders)
    truth_path = os.path.join(CODING_DIR, "coding_truth.csv")
    with open(truth_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["row_id", "response", "true_sentiment", "true_adoption_stage",
                      "true_main_barrier", "pred_sentiment", "pred_adoption_stage",
                      "pred_main_barrier", "error_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in selected:
            writer.writerow({**r, "error_count": count_errors(r)})

    # Generate HTML with responses embedded
    response_json = json.dumps(
        [{"row_id": int(r["row_id"]), "response": r["response"]} for r in selected],
        ensure_ascii=False
    )
    html = (HTML_TEMPLATE
            .replace("__RESPONSES_JSON__", response_json)
            .replace("__N__", str(len(selected))))
    html_path = os.path.join(CODING_DIR, "coder.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    err_counts = Counter(count_errors(r) for r in selected)
    print(f"Coding task ready — {len(selected)} responses selected")
    print(f"  Error distribution: {dict(sorted(err_counts.items()))}  "
          f"({err_counts[0]} correct, {sum(v for k,v in err_counts.items() if k>0)} with LLM errors)")
    print(f"\nFiles created:")
    print(f"  {html_path}     <-- share this with human coders")
    print(f"  {truth_path}  <-- keep private (contains ground truth)")
    print(f"\nNext steps:")
    print(f"  1. Share  coding_task/coder.html  with 2–3 colleagues")
    print(f"  2. Ask each to open it in a browser, complete all {len(selected)} responses,")
    print(f"     click 'Download CSV', and send you the file")
    print(f"  3. Save received files as  coding_task/coder_<name>.csv")
    print(f"  4. Run:  python human_coding_analysis.py")


if __name__ == "__main__":
    main()
