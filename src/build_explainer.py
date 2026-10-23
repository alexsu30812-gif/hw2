"""Create the student's Chinese reading guide from the measured report payload."""
from pathlib import Path
from html import escape
import json

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    data = json.loads((root / 'results/report_data.json').read_text())
    a = data['basic_a']
    selected = a['selected_configuration']
    rows = []
    reference = a['real_baseline']
    rows.append(f"<tr><td>真實資料參考（{reference['n']} 首）</td><td>{reference['H4']['mean']:.4f}</td><td>{reference['GS']['mean']:.4f}</td></tr>")
    for run in a['configurations']:
        label = escape(run['id']) + ('（提交設定）' if run['id'] == selected else '')
        rows.append(f"<tr><td>{label}</td><td>{run['metrics']['H4']['mean']:.4f}</td><td>{run['metrics']['GS']['mean']:.4f}</td></tr>")
    choices = ', '.join(f"{row['prompt_id']}：{row['configuration_id']}"
                        for row in data['basic_b']['results'] if row.get('is_best'))
    bonus = data['bonus']
    bonus_rows = []
    for name in ('CA', 'OA_P', 'OA_V', 'OA_D', 'OA_IOI'):
        values = [row['metrics'][name] for row in bonus['results']]
        values = [v['mean'] if isinstance(v, dict) else v for v in values]
        bonus_rows.append(f'<tr><td>{name}</td><td>{sum(values)/len(values):.4f}</td></tr>')
    content = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>HW2 中文原理解說</title>
<style>body{font:17px/1.8 system-ui,sans-serif;max-width:960px;padding:24px;margin:auto;color:#222}h1,h2{line-height:1.4}h2{margin-top:2em}table{border-collapse:collapse;width:100%;margin:1em 0}td,th{border-bottom:1px solid #ccc;text-align:left;padding:8px}code{background:#eee;padding:2px 5px}a{color:#1456a0}summary{cursor:pointer;font-weight:bold}details{border-top:1px solid #ccc;padding:14px 0}.flow{padding:16px;background:#f3f3f3}.note{border-left:3px solid #777;padding-left:14px}img{width:100%}</style>
<h1>HW2：這份作業做了什麼？</h1>
<p>學號 r14942154。這份是中文閱讀說明；正式報告是 <a href="r14942154_report.pdf">英文 PDF</a>。<a href="listening.html">開啟 39 組成果試聽</a>。</p>
<h2>一、三個任務</h2><table><tr><th>任務</th><th>輸入、輸出</th><th>提交數量</th></tr>
<tr><td>Basic a</td><td>從開頭產生完整 32 小節</td><td>選定設定的 20 組 MIDI／WAV</td></tr>
<tr><td>Basic b</td><td>保留指定的 8 小節，再接上 24 小節</td><td>3 首指定曲 × 3 設定，共 9 組</td></tr>
<tr><td>Bonus</td><td>讀取 16 小節旋律，產生鋼琴伴奏，保留原旋律</td><td>10 組 MIDI／WAV</td></tr></table>
<p>Basic a 實際比較了五種設定，每種 20 首，共 100 首；提交其中一組。訓練只用課程提供的 896 首，權重從隨機初始化開始。指定曲和伴奏答案沒有拿去訓練。</p>
<h2>二、模型如何學會產生音樂？</h2>
<div class="flow">MIDI 音符＋beat_midi.txt → 音樂 token → Transformer 預測下一個 token → 依機率取樣 → 還原 MIDI → 合成 WAV</div>
<details open><summary>1. 把音符變成一串符號</summary><p>例如「第 3 小節、第 2 拍附近，彈 C4、中等音量、維持半拍」拆成 Bar、位置、音高、音量、時長等符號。每個註記拍切成 8 格，總詞表 683 個。這樣模型能像預測句子的下一個字一樣，預測接下來的音樂事件。</p><p>助教提醒原 MIDI 拍號可能錯誤，所以小節邊界從 beat_midi.txt 計算。輸出把一個註記拍當作一個四分音符：六個註記拍寫成 6/4；這保留時間和小節數，並不代表找回原曲究竟是 6/8 還是其他記譜方式。</p></details>
<details open><summary>2. 用 Transformer 預測下一個符號</summary><p>模型有 6 層、每層寬度 384、6 個注意力頭。因果遮罩讓它只能看前面，不能偷看後面。訓練時提供真實前文，要求下一個 token 的機率變高；交叉熵越低，代表在這組資料上預測得越準。這不等於生成的音樂一定好聽。</p><p>先把歌曲分成 851 首訓練、45 首驗證來選設定；最後再用全部 896 首繼續訓練提交模型。因此早期 45 首的驗證結果只適用於早期模型，不能宣稱它們仍是最終模型沒看過的資料。</p></details>
<details open><summary>3. 取樣讓模型真的寫出新曲子</summary><p>Temperature 降低時，模型更偏向高機率音符，容易穩定，也可能過度重複。提高時選擇更多，但可能鬆散。Top-k 保留機率最高的 k 個候選；top-p 保留累積機率達到 p 的候選。本作業比較 T=0.8、1.0、1.2（p=0.95）和 k=16、32。</p><p>解碼器另外限制符號順序、合法音高、小節長度與每小節音符上限，避免無法解析的 MIDI。KV cache 把前文計算存起來，減少每個音符重算整段的成本；超出上下文容量時，從完整小節邊界保留最近的前文。</p></details>
<details><summary>4. 續寫與伴奏怎麼做？</summary><p>續寫先讀 8 小節前文，再用 Basic a 同一個模型產生 24 小節。匯出時放回原始前文音符，避免量化改掉指定曲。伴奏模型用同一詞表加上 MELODY／PIANO 軌道符號，每小節先讀指定旋律再產生鋼琴。它能讀當前小節和過去的旋律，不是事先看完整首未來旋律；伴奏答案與和弦標註只供評估。</p></details>
<h2>三、如何判斷結果？</h2>
<p>H4 是四小節內音高類別的熵，可以想成用了多少不同的音、分布是否集中。GS 比較小節之間的起音節奏。兩者都要與真實資料比較，並非越大越好；隨機音高可能把 H4 拉高，空白或重複節奏也可能把 GS 拉高。</p>
<table><tr><th>資料／設定</th><th>平均 H4</th><th>平均 GS</th></tr>__A_ROWS__</table>
<p>Basic a 依事先訂好的規則，將 H4／GS 與真實參考的差距各除以參考資料標準差，再取平均，選擇差距最小的設定。另檢查音符密度、空白小節和重複情況。這個規則是本實驗的選法，助教沒有公布指標換算成作業分數的公式。</p>
<p>Basic b 每首選定設定：__CHOICES__。選擇依據是第 8、9 小節交界附近的密度、平均音高與音高類別變化，屬於局部統計；沒有假稱做過人工盲聽。</p>
<table><tr><th>Bonus 指標</th><th>十首平均</th></tr>__BONUS_ROWS__</table>
<p>CA 檢查伴奏與標註和弦的一致性。OA 比較生成伴奏與答案在音高類別、音量、時長、起音間隔的分布重疊。OA 不保留完整音符順序，音高類別還會忽略八度，因此高分仍不能保證音樂合理。和弦一致性是這份模型仍需改善的部分。</p>
<h2>四、額外實驗的意義</h2><ul>
<li>不同量化解析度：比較時間誤差與詞表大小，說明為何選每拍 8 格。這是重建實驗，沒有把三種解析度都重新訓練。</li>
<li>不同訓練階段：比較同一組驗證窗口的下一符號預測，確認音高等內容是否進步；同時檢查自由生成結果。</li>
<li>2048／4096 上下文：從相同父模型，各訓練約 400 萬非填充目標 token。較長前文在這個預算下沒有得到較低驗證誤差，所以選 2048。</li>
<li>移除旋律輸入：固定 T=1，保留小節與速度，只移除提供給模型的旋律音符，再比較伴奏。這會改變輸入分布和上下文長度，不能當作音樂品質的純因果證明。</li>
<li>真實參考敏感度：用全部可評估歌曲再算一次，檢查 45 首參考在 H4／GS／密度上是否偏離整體。</li>
<li>逐檔檢查：小節數、指定旋律保留、MIDI／WAV 對應、音訊非靜音且無削波，以及檔案雜湊。零個完全相同片段不代表已證明沒有近似記憶。</li></ul>
<img src="figures/sampling_distributions.png" alt="H4 與 GS 各設定的分布">
<h2>五、繳交方式與限制</h2>
<p>NTU COOL 上傳 <code>r14942154_report.pdf</code>，作業留言放 <a href="https://github.com/alexsu30812-gif/hw2">公開 GitHub 連結</a>。程式、兩個自己的模型、README、requirements、39 組 MIDI／WAV 都放在成果資料夾。訓練資料、tokenized 訓練陣列與快取不在提交包中。</p>
<p class="note">你選擇用 GitHub 公開成果。作業原文寫 cloud drive，目前沒有助教確認 GitHub 可替代的紀錄；這項宿主要求仍未證實。報告與提交檢查會保留此說明。AI 協助已在正式報告揭露，數字都來自實際執行，沒有保證成績。</p>
</html>'''
    content = content.replace('__A_ROWS__', ''.join(rows)).replace('__CHOICES__', escape(choices)).replace('__BONUS_ROWS__', ''.join(bonus_rows))
    target = root / 'HW2_說明.html'
    target.write_text(content, encoding='utf-8')
    return target


if __name__ == '__main__':
    print(build())
