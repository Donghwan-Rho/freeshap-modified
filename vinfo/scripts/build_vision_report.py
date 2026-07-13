# -*- coding: utf-8 -*-
"""Bundle the VISION (cifar10) Shapley comparison figures + explanations into one PDF.
Same 5-part structure as the NLP report, adapted to a rank/d sweep (fixed lambda=1e-2):
  A data-selection accuracy (txt) / (1) fidelity heatmap / (2) abs-diff bars /
  (3) error dist / (4) ranking preservation.  Figures from plot_acc_vision.py (A)
  and shap_compare_vision.py (1-4) into report_figs_vision/.
Output: <vinfo>/shap_value_comparison_report_vision.pdf
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Image, PageBreak)
import xml.sax.saxutils as su

pdfmetrics.registerFont(UnicodeCIDFont('HYSMyeongJo-Medium'))
pdfmetrics.registerFont(UnicodeCIDFont('HYGothic-Medium'))
KO = 'HYSMyeongJo-Medium'; KOG = 'HYGothic-Medium'
INK = colors.HexColor('#1a1a1a'); ACC = colors.HexColor('#0b5394'); GREEN = colors.HexColor('#1a9850')
RED = colors.HexColor('#b2182b'); GREY = colors.HexColor('#555555')

st = {
 'title': ParagraphStyle('t', fontName=KOG, fontSize=18, leading=24, textColor=ACC, spaceAfter=8),
 'h':     ParagraphStyle('h', fontName=KOG, fontSize=14, leading=19, textColor=ACC, spaceBefore=4, spaceAfter=6),
 'sh':    ParagraphStyle('sh', fontName=KOG, fontSize=11.5, leading=16, textColor=ACC, spaceBefore=2, spaceAfter=4),
 'body':  ParagraphStyle('b', fontName=KO, fontSize=10, leading=15.5, textColor=INK, spaceAfter=5, alignment=4),
 'small': ParagraphStyle('s', fontName=KO, fontSize=8.7, leading=12.5, textColor=GREY, spaceAfter=4),
 'cap':   ParagraphStyle('c', fontName=KO, fontSize=8.3, leading=11, textColor=GREY, spaceBefore=3, spaceAfter=2, alignment=1),
}
def esc(s): return su.escape(str(s))
def g(t): return f'<font name="{KOG}">{esc(t)}</font>'
def green(t): return f'<font name="{KOG}" color="#1a9850">{esc(t)}</font>'
def red(t): return f'<font name="{KOG}" color="#b2182b">{esc(t)}</font>'

story = []
def P(t, s='body'): story.append(Paragraph(t, st[s]))
def SP(h=5): story.append(Spacer(1, h))

V = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../vinfo
OUT = os.path.join(V, 'report_figs_vision')
def FIG(name, cap, maxw=172*mm, maxh=210*mm):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        story.append(Paragraph('(figure not found: ' + name + ')', st['small'])); return
    iw, ih = ImageReader(path).getSize(); r = iw / ih
    w = maxw; h = w / r
    if h > maxh: h = maxh; w = h * r
    story.append(Image(path, width=w, height=h)); story.append(Paragraph(cap, st['cap']))

# ===================== overview =====================
P('Shapley Value 비교 리포트 (Vision / CIFAR-10): inv vs eigen / nystrom', 'title')
P('원본(FreeShap, ' + g('inv') + ' = full eNTK)으로 구한 점별 Shapley 값을 ' + g('기준') +
  '으로, 근사 방법 ' + g('eigen') + '•' + g('nystrom') + '이 ' + g('같은 데이터 포인트') +
  '에서 값/순위를 얼마나 다르게 추정하는지 본다. NLP 리포트가 λ를 스윕한 것과 달리, '
  'vision 실험(n01_vision.sh)은 ' + g('λ=1e-2 고정, rank/d를 스윕') + '한다.')
SP()
P('실험 설정', 'h')
P('• 데이터셋: cifar10 (ResNet eNTK)   • seed: 2024, 2025, 2026   • tmc=500', 'small')
P('• ' + g('sweep') + ': eigen rank ∈ {1,5,10,15,20,25,30}%, nystrom d ∈ {1,5,10,15,20,25,30}% (λ=1e-2 고정)', 'small')
P('• 점별 Shapley 값 = dv_result[:,1,:].sum(axis=1) / 같은 seed → sampled_idx 동일 → 같은 위치=같은 포인트', 'small')
P('• data_selection: inv•eigen num5000, nystrom num2500(진행 중) / shapley pkl: 전부 num5000', 'small')
SP(8)
P('구조: 5개 분석 파트', 'h')
P('• ' + g('A 데이터선택 정확도') + '(txt) / ' + g('① 충실도 히트맵') + ' / ' + g('② 평균•최대 오차막대') +
  ' / ' + g('③ 점별 오차분포') + ' / ' + g('④ 순위 보존도') + ' (①~④는 shapley pkl)', 'small')
P('• 각 파트는 ' + g('행=seed, 열/색=rank•d 스윕') + '으로 본다. ' + g('결과 없는 칸은 빈칸') +
  ' (seed2026 전부, nystrom data_selection은 진행 중이라 Part A에서 빈칸).', 'small')
SP(6)
P('※ A는 data_selection 결과(txt, git 추적)에서, ①~④는 shapley 결과(pkl, 서버 로컬•gitignore)에서 계산.', 'small')
story.append(PageBreak())

# ===================== Part A =====================
P('A. 데이터 선택 정확도 (랭킹 기반)', 'h')
P('데이터 선택은 ' + g('eigen/nystrom으로 근사한 Shapley 랭킹') + '으로 상위 k%를 고르되, '
  '정확도 측정은 반드시 ' + g('full eNTK(inv) predictor') + '로 한다. '
  'x=선택 비율, y=검증 정확도. ' + g('검은 굵은 선=inv 랭킹(oracle)') + ', 색선=각 rank/d의 근사 랭킹 — '
  + g('모두 full eNTK로 평가') + '. 근사 곡선이 검은 선에 붙을수록 좋은 랭킹.')
P('예측 txt에는 같은 근사 랭킹을 ' + g('inv mode(full eNTK로 평가)') + '와 ' + g('eigen/nystrom mode(저랭크 predictor로 평가)') +
  ' 두 가지로 저장한다. 후자는 selection≈rank d% 근처에서 정확도가 인위적으로 무너지는 ' + g('아티팩트') +
  '가 있어, 리포트는 ' + g('inv mode 곡선만') + ' 쓴다.', 'small')
SP(4)
FIG('acc_eigen.png', 'eigen rank 스윕 — 행=seed. 근사 랭킹을 full eNTK로 평가 → 곡선이 inv oracle을 매끄럽게 추종', maxh=224*mm)
story.append(PageBreak())
FIG('acc_nystrom.png', 'nystrom d 스윕 — 행=seed. (data_selection nystrom 진행 중이면 빈칸)', maxh=224*mm)
story.append(PageBreak())
FIG('acc_matched.png', 'matched — 같은 rank/d에서 inv vs eigen vs nystrom 직접 비교 (행=seed, 열=rank/d %)', maxh=150*mm)
story.append(PageBreak())

# ===================== Part ① =====================
P('① 충실도 히트맵 (correlation & mean|Δ|)', 'h')
P('한 칸 = (seed, 방법). ' + g('위') + '=inv와의 상관(1=동일, 0=무관), ' + g('아래') +
  '=점별 평균 절대오차 mean|approx-inv|. ' + green('초록=inv와 가까움') + ', 빨강=멀다. '
  '왼쪽 그룹=eigen rank, 오른쪽=nystrom d (검은 세로선으로 구분).')
P('eigen은 rank가 커질수록 상관↑(0.5→0.8)•오차 낮음. nystrom은 ' + g('d 5%부터 값이 폭발') +
  '(상관≈0.06, mean|Δ|≈1.8)하는 경향.', 'small')
SP(4)
FIG('heatmap.png', 'correlation(위)•mean|Δ|(아래) — 초록=inv와 가까움', maxh=170*mm)
story.append(PageBreak())

# ===================== Part ② =====================
P('② 평균•최대 오차 막대 (log y)', 'h')
P('각 칸=(seed, 방법군). ' + g('막대=평균오차 mean|Δ|') + ', ' + g('위 tick=최대오차 max|Δ|') +
  '. y=log, ' + red('빨간선') + '=정상 sv 스케일(≈1). 막대가 빨간선을 넘으면 사실상 폭발. '
  '왼쪽=eigen, 오른쪽=nystrom.')
SP(4)
FIG('absdiff.png', '막대=mean|Δ|, tick=max|Δ| (log y; 빨간선≈정상 스케일) — 행=seed, 좌=eigen•우=nystrom', maxh=210*mm)
story.append(PageBreak())

# ===================== Part ③ =====================
P('③ 점별 오차 분포', 'h')
P('점별 오차 = (approx 값 - inv 값)의 히스토그램. x=오차(0이면 inv와 일치). ' + g('|오차|>1은 ±1로 클리핑') +
  '해 양끝에 쌓음(좌상단에 클리핑 비율). ' + green('0 중심으로 좁고 뾰족할수록 inv에 충실') +
  '. 폭발 시 0 근처가 비고 양끝(±1)에 스파이크.')
SP(4)
FIG('errordist.png', '(approx - inv) 점별 오차분포 — ±1 클리핑, 행=seed, 좌=eigen•우=nystrom', maxh=210*mm)
story.append(PageBreak())

# ===================== Part ④ =====================
P('④ 순위(ranking) 보존도 — Spearman & top-5% 겹침', 'h')
P('①②③은 값의 유사도, 여기는 ' + g('순위가 inv와 같은지') + '. ' + g('위') + '=Spearman 순위상관(1=같은 순서), '
  + g('아래') + '=inv top-5% 데이터와의 집합 겹침(무작위면 ~0.05). ' + green('초록=순위 잘 보존') + '.')
P('값이 터진 nystrom 칸은 순위도 무너지는 경향(Spearman≈0, 겹침≈무작위). eigen은 값•순위 모두 잘 보존.', 'small')
SP(4)
FIG('ranking.png', 'Spearman(위)•top-5% 겹침(아래) — 초록=순위 보존', maxh=170*mm)
story.append(PageBreak())

# ===================== takeaway =====================
P('정리 & 시사점', 'h')
P('• ' + green('eigen') + '은 Shapley ' + g('값과 순위 둘 다') + ' inv에 충실하고, rank가 커질수록 더 좋아진다 → 값•순위 기반 분석 모두 안전.')
P('• ' + red('nystrom') + '은 ' + g('d 5% 이상에서 값이 폭발') + '하고 순위도 inv와 무관해지는 경향(d 1%만 그나마 안정). '
  '데이터 선택 정확도(A)가 그래도 괜찮아 보이는 건 무작위에 가까운 선택도 정확도가 그럭저럭 나오기 때문.')
P('• NLP과 달리 vision은 ' + g('λ 고정•rank/d 스윕') + '이라, 여기서는 ' + g('rank/d가 클수록 eigen이 안정') +
  '되는지, nystrom이 어느 d부터 터지는지를 본다.', 'small')
SP(8)
P('※ 데이터 추가되면 shap_compare_vision.py → plot_acc_vision.py → build_vision_report.py 순으로 재생성.', 'small')

def footer(c, d):
    c.saveState(); c.setFont(KO, 8); c.setFillColor(GREY)
    c.drawString(20*mm, 12*mm, 'Shapley value 비교 (vision/cifar10): inv vs eigen / nystrom')
    c.drawRightString(190*mm, 12*mm, f'p.{d.page}'); c.restoreState()

doc = BaseDocTemplate(f'{V}/shap_value_comparison_report_vision.pdf', pagesize=A4,
                      leftMargin=20*mm, rightMargin=20*mm, topMargin=16*mm, bottomMargin=18*mm,
                      title='Shapley value comparison report (vision/cifar10)')
fr = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='n')
doc.addPageTemplates([PageTemplate(id='m', frames=[fr], onPage=footer)])
doc.build(story)
print('saved:', f'{V}/shap_value_comparison_report_vision.pdf')
