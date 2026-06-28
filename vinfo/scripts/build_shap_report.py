# -*- coding: utf-8 -*-
"""Bundle the Shapley-value comparison figures + explanations into one PDF.
Structure: 5 analysis parts (A 데이터선택정확도 / ① 충실도히트맵 / ② 오차막대 /
③ 오차분포 / ④ 순위보존), each shown for 4 comparison settings (s1..s4).
Figures are produced by shap_compare.py (①~④) and plot_eig_nys.py (A) into report_figs/.
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
                                Spacer, Image, PageBreak, HRFlowable)
import xml.sax.saxutils as su

pdfmetrics.registerFont(UnicodeCIDFont('HYSMyeongJo-Medium'))
pdfmetrics.registerFont(UnicodeCIDFont('HYGothic-Medium'))
KO='HYSMyeongJo-Medium'; KOG='HYGothic-Medium'
INK=colors.HexColor('#1a1a1a'); ACC=colors.HexColor('#0b5394'); GREEN=colors.HexColor('#1a9850')
RED=colors.HexColor('#b2182b'); GREY=colors.HexColor('#555555'); EXBG=colors.HexColor('#fdeef0')

st={
 'title':ParagraphStyle('t',fontName=KOG,fontSize=18,leading=24,textColor=ACC,spaceAfter=8),
 'h':ParagraphStyle('h',fontName=KOG,fontSize=14,leading=19,textColor=ACC,spaceBefore=4,spaceAfter=6),
 'sh':ParagraphStyle('sh',fontName=KOG,fontSize=11.5,leading=16,textColor=colors.HexColor('#0b5394'),spaceBefore=2,spaceAfter=4),
 'body':ParagraphStyle('b',fontName=KO,fontSize=10,leading=15.5,textColor=INK,spaceAfter=5,alignment=4),
 'small':ParagraphStyle('s',fontName=KO,fontSize=8.7,leading=12.5,textColor=GREY,spaceAfter=4),
 'ex':ParagraphStyle('e',fontName=KO,fontSize=9.6,leading=15,textColor=INK,backColor=EXBG,
                     borderColor=colors.HexColor('#e3b9c0'),borderWidth=0.6,borderPadding=(6,7,6,7),
                     spaceBefore=3,spaceAfter=6),
 'cap':ParagraphStyle('c',fontName=KO,fontSize=8.3,leading=11,textColor=GREY,spaceBefore=3,spaceAfter=2,alignment=1),
}
def esc(s): return su.escape(str(s))
def g(t): return f'<font name="{KOG}">{esc(t)}</font>'
def green(t): return f'<font name="{KOG}" color="#1a9850">{esc(t)}</font>'
def red(t): return f'<font name="{KOG}" color="#b2182b">{esc(t)}</font>'

story=[]
def P(t,s='body'): story.append(Paragraph(t,st[s]))
def SP(h=5): story.append(Spacer(1,h))
def FIG(path,cap,maxw=165*mm,maxh=158*mm):
    if not os.path.exists(path):
        story.append(Paragraph('(figure not found: '+os.path.basename(path)+')',st['small'])); return
    iw,ih=ImageReader(path).getSize(); r=iw/ih
    w=maxw; h=w/r
    if h>maxh: h=maxh; w=h*r
    story.append(Image(path,width=w,height=h)); story.append(Paragraph(cap,st['cap']))

V=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../vinfo (서버 무관)
OUT=os.path.join(V,'report_figs')

# 4 comparison settings (must match shap_compare.py / plot_eig_nys.py tags)
SETTINGS=[
 ('s1','세팅 1) eigen λ=1e-2  vs  nystrom λ=1e-1~1e-6'),
 ('s2','세팅 2) eigen 내부 λ=1e-1~1e-6'),
 ('s3','세팅 3) nystrom 내부 λ=1e-1~1e-6'),
 ('s4','세팅 4) matched: λ별 eigen vs nystrom (인접 열)'),
]
DS_A=[('sst2',5000),('mnli',5000),('ag_news',5000),('mr',5000),('qqp',5000),('rte',2490),('mrpc',3668)]

def part(part_label, prefix, intro, cap, maxw=172*mm, maxh=152*mm,
         matched_perds=False, matched_maxw=172*mm, matched_maxh=210*mm):
    """한 분석 파트: 설명 1페이지 + 세팅 4개(각 1페이지).
    matched_perds=True면 s4(matched)를 데이터셋별 (행=λ,열=rank) 페이지로 전개."""
    P(part_label,'h')
    for t,s in intro: P(t,s)
    P('아래로 ' + g('세팅 4가지(s1~s4)') + '를 같은 분석으로 차례로 본다. '
      + g('결과 없는 칸/방법은 빈칸') + '.', 'small')
    story.append(PageBreak())
    for stag,slabel in SETTINGS:
        if stag=='s4' and matched_perds:
            P(f'{part_label}','h'); P('세팅 4) matched — 데이터셋별 (행=λ 1e-1~1e-6, 열=rank 1/5/20%)','sh')
            P('각 칸 = 같은 λ에서 ' + g('eigen(검정) vs nystrom(빨강)') + ' 직접 비교.', 'small')
            story.append(PageBreak())
            for ds,num in DS_A:
                P(f'{part_label} — matched — {ds}','sh')
                FIG(f'{OUT}/{prefix}_s4_{ds}.png', f'{ds} (num{num}): eigen vs nystrom at same λ — 행=λ, 열=rank',
                    maxw=matched_maxw, maxh=matched_maxh)
                story.append(PageBreak())
        else:
            P(f'{part_label}', 'h'); P(slabel,'sh')
            FIG(f'{OUT}/{prefix}_{stag}.png', f'{slabel} — {cap}', maxw=maxw, maxh=maxh)
            story.append(PageBreak())

# ===================== overview =====================
P('Shapley Value 비교 리포트: inv vs eigen / nystrom','title')
P('원본(FreeShap, ' + g('inv') + ' = full eNTK)으로 구한 점별 Shapley 값을 ' + g('기준') +
  '으로, 근사 방법 ' + g('eigen') + '·' + g('nystrom') + '이 ' + g('같은 데이터 포인트') +
  '에서 값/순위를 얼마나 다르게 추정하는지 본다.')
SP()
P('실험 설정','h')
P('· 데이터셋 7개: sst2, mnli, ag_news, mr, qqp, rte, mrpc &nbsp;&nbsp;· approx rank: 1%, 5%, 20% &nbsp;&nbsp;· seed=2024, tmc=500','small')
P('· λ sweep: 1e-1 … 1e-6 / 점별 Shapley 값 = dv_result[:,1,:].sum(axis=1)','small')
P('· 같은 seed라 모든 방법의 sampled_idx 동일 → 같은 위치 = 같은 데이터 포인트','small')
SP(8)
P('구조: 5개 분석 파트 × 4개 비교 세팅','h')
P('· ' + g('분석 파트') + ': A 데이터선택 정확도 / ① 충실도 히트맵 / ② 평균·최대 오차막대 / ③ 점별 오차분포 / ④ 순위 보존도','small')
P('· ' + g('비교 세팅') + ':','small')
P('&nbsp;&nbsp;1) eigen λ=1e-2 vs nystrom λ=1e-1~1e-6 &nbsp;&nbsp; 2) eigen 내부(λ=1e-1~1e-6)','small')
P('&nbsp;&nbsp;3) nystrom 내부(λ=1e-1~1e-6) &nbsp;&nbsp; 4) matched: λ별 eigen vs nystrom 인접 열','small')
SP(6)
P('※ A(데이터선택 정확도)는 data_selection 결과(txt)에서, ①~④는 shapley 결과(pkl)에서 계산. '
  'pkl은 서버 로컬에만 있어(gitignore), 그 서버에 없는 방법(예: nystrom)은 ①~④에서 빈칸일 수 있다.', 'small')
story.append(PageBreak())

# ===================== Part A =====================
part('A. 데이터 선택 정확도 (랭킹 기반)','acc',
 [(('각 방법의 Shapley 랭킹으로 상위 k%를 고른 뒤 ' + g('full eNTK(inv)') + '로 정확도 측정. '
    'x=선택 비율, y=검증 정확도. 곡선이 위일수록 더 좋은 데이터를 골랐다는 뜻.'),'body'),
  (('대부분 칸에서 곡선이 겹친다 = 랭킹 기반 선택 정확도는 방법 간 차이가 작다.'),'body')],
 '곡선 위일수록 좋은 랭킹', maxw=152*mm, maxh=188*mm, matched_perds=True)

# ===================== Part ① =====================
part('① 충실도 히트맵 (correlation & mean|Δ|)','heatmap',
 [(('한 칸 = (데이터셋, rank) × 방법. ' + g('왼쪽') + '=inv와의 상관(1=동일, 0=무관), '
    + g('오른쪽') + '=점별 평균 절대오차 mean|approx−inv|. ' + green('초록=inv와 가까움') + ', 빨강=멀다.'),'body'),
  (('eigen 열은 대체로 초록, nystrom은 ' + g('작은 λ·rank 5%') + '에서 새빨갛게(값 폭발) 변하는 경향.'),'body')],
 'correlation(좌)·mean|Δ|(우) — 초록=inv와 가까움', maxw=176*mm, maxh=150*mm)

# ===================== Part ② =====================
part('② 평균·최대 오차 막대 (log y)','absdiff',
 [(('각 칸=(데이터셋,rank)의 방법별 막대. ' + g('막대=평균오차 mean|Δ|') + ', ' + g('위 tick=최대오차 max|Δ|') +
    '. y=log, ' + red('빨간선') + '=정상 sv 스케일(≈1). 막대가 빨간선을 넘으면 사실상 폭발.'),'body')],
 '막대=mean|Δ|, tick=max|Δ| (log y; 빨간선≈정상 스케일)', maxw=152*mm, maxh=188*mm)

# ===================== Part ③ =====================
part('③ 점별 오차 분포','errordist',
 [(('점별 오차 = (approx 값 − inv 값)의 히스토그램. x=오차(0이면 inv와 일치). ' +
    g('|오차|>1은 ±1로 클리핑') + '해 양끝에 쌓음(좌상단에 클리핑 비율).'),'body'),
  ((green('0 중심으로 좁고 뾰족할수록 inv에 충실') + '. 폭발 시 0 근처가 비고 양끝(±1)에 스파이크.'),'body')],
 '(approx − inv) 점별 오차분포 — ±1 클리핑', maxw=152*mm, maxh=188*mm, matched_perds=True)

# ===================== Part ④ =====================
part('④ 순위(ranking) 보존도 — Spearman & top-5% 겹침','ranking',
 [(('①②③은 값의 유사도, 여기는 ' + g('순위가 inv와 같은지') + '. ' + g('왼쪽') + '=Spearman 순위상관(1=같은 순서), '
    + g('오른쪽') + '=inv top-5% 데이터와의 집합 겹침(무작위면 ~0.05). ' + green('초록=순위 잘 보존') + '.'),'body'),
  (('값이 터진 칸은 순위도 무너지는 경향(Spearman≈0, 겹침≈무작위). eigen은 값·순위 모두 잘 보존.'),'body')],
 'Spearman(좌)·top-5% 겹침(우) — 초록=순위 보존', maxw=176*mm, maxh=150*mm)

# ===================== takeaway =====================
P('정리 & 시사점','h')
P('· ' + green('eigen') + '은 Shapley ' + g('값과 순위 둘 다') + ' inv에 충실 → 값·순위 기반 분석 모두 안전.')
P('· ' + red('nystrom') + '은 mid-rank·작은 λ에서 값이 폭발하고 순위도 inv와 무관해지는 경향. '
  '데이터 선택 정확도(A)가 그래도 괜찮아 보이는 건 무작위에 가까운 선택도 정확도가 그럭저럭 나오기 때문.')
P('· 세팅 2(eigen 내부)/3(nystrom 내부)은 ' + g('각 방법이 어떤 λ에서 안정적인지') + ', '
  '세팅 4(matched)는 ' + g('같은 λ에서 eigen이 nystrom보다 얼마나 나은지') + '를 직접 보여준다.', 'small')
SP(8)
P('※ 그림은 seed=2024, tmc=500 기준. 데이터 추가되면 shap_compare.py → plot_eig_nys.py → build_shap_report.py 순으로 재생성.', 'small')

def footer(c,d):
    c.saveState(); c.setFont(KO,8); c.setFillColor(GREY)
    c.drawString(20*mm,12*mm,'Shapley value 비교: inv vs eigen / nystrom')
    c.drawRightString(190*mm,12*mm,f'p.{d.page}'); c.restoreState()

doc=BaseDocTemplate(f'{V}/shap_value_comparison_report.pdf',pagesize=A4,
                    leftMargin=20*mm,rightMargin=20*mm,topMargin=16*mm,bottomMargin=18*mm,
                    title='Shapley value comparison report')
fr=Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,id='n')
doc.addPageTemplates([PageTemplate(id='m',frames=[fr],onPage=footer)])
doc.build(story)
print('saved:',f'{V}/shap_value_comparison_report.pdf')
