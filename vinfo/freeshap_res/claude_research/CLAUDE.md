# FreeSHAP 연구 자동화

## 컨텍스트
- 노드: node04, conda env: freeshap
- 작업 루트: /extdata1/donghwan/freeshap/vinfo/freeshap_res/claude_research
- 상위 repo (읽기 전용 참조): /extdata1/donghwan/freeshap
- 기존 실험 스크립트 예: ../../n05_0.sh
- GPU: nvidia-smi로 빈 것 자동 선택
- 모델: claude-opus-4-7 (Max 5x 플랜)

## 한 iteration 표준 절차
1. 사용자가 state/latest_directive.md 에 지시를 넣는다 (또는 채팅으로 줌)
2. @planner → state/iteration_NN/plan.md
3. @critic → critique.md
4. @planner 재호출 → plan.md 수정
5. @executor → experiments/iter_NN_exp_K.sh 생성 → nohup 백그라운드 실행
6. 실험 종료 대기 (executor가 주기적 tail 확인)
7. @critic → 결과 분석 → critique.md 업데이트
8. `python scripts/make_report.py NN` → reports/iter_NN.pdf (full archive, all rounds)
   `python scripts/make_summary_report.py NN` → reports/iter_NN_summary.pdf
   (reading copy: final plan + final critique + next_directions + bibliography only)
   반드시 두 스크립트 모두 실행할 것.
9. 사용자 피드백 대기

## 절대 금지
- foreground로 학습 실행
- rm -rf, git push --force
- /extdata1/donghwan/freeshap 하위 (claude_research 외) 파일 수정
- ANTHROPIC_API_KEY 설정

## Language policy
- 사용자 질문·지시: 한국어.
- 모든 영구 산출물 (plan.md, critique.md, next_directions.md, bibliography.md,
  PDF 리포트, 채팅 상태 보고) 은 **한국어로 작성**.
- 예외: 다음 항목은 영어 원문 유지 (번역 금지).
  - 학술 용어: Shapley value, rank-r approximation, eigenvalue, effective
    dimension, NTK, TMC, Nyström, gradient flow, spectral filtering,
    implicit regularization, bias-variance, PAC-Bayes, stability 등.
  - 수식·기호·상수 (ρ, λ, d_eff, LC(r), O(·) 표기 등).
  - 논문 제목·저자명·venue·URL·arXiv ID.
  - 파일·함수·변수·명령어명.
  - 페이지/수식/§ 번호, [B#] 식 bibliography 레퍼런스.
- 문체: 논문 비평/분석에 맞는 건조하고 정밀한 한국어. "~인 것 같다" 같은
  hedging 남용 금지. 주장은 명확히, 근거는 페이지·수식 번호로.
- 코드·코드 주석·commit message·shell script·log message 는 **영어** 유지
  (운영 호환성).
- Sub-agent (planner, critic, executor) 도 이 정책을 동일하게 따른다.
  한국어 지시를 받아도, 이 규칙에 따라 한국어/영어를 혼용해 작성한다.

## Writing style policy (문체)
모든 영구 산출물 (plan*.md, critique*.md, next_directions.md, bibliography 설명문,
PDF 리포트) 은 아래 문체를 따른다. 채팅 상태 보고는 예외.

- **톤**: 대학원생이 지도교수와의 1:1 미팅에서 설명하듯이 서술.
  "이 부분 Prop 9.5 bound 를 보면, 저자들은 ρ=1e-2 일 때 ρ² 로 나누는
  단계에서 10⁴ 배 blow-up 이 생기는데, 제 생각엔 Caponnetto–De Vito 의
  effective dimension d_eff(ρ) 를 도입하면 d_eff/ρ 로 완화할 수 있을 것
  같습니다. 다만 조건이 붙어서..." 같은 형식.
- **서술 우선, bullet 보조**: 한 sub-agent 산출물은 긴 서술 문단 + 필요한
  표/목록 혼합 (대략 8 : 2). 진짜 병렬 항목 (여러 데이터셋 비교, 수정사항
  항목 열거 등) 에서만 bullet/표 사용. 전부 목록 나열식 금지.
- **논리 흐름 강제**: "무엇을 관찰했는가 → 왜 중요한가 → 제 생각 / 제안 →
  그러나 걸리는 점 → 검증 방법" 의 흐름을 문단 안에서 자연스럽게 구성.
- **정확성은 포기하지 않음**:
  - 페이지 번호·수식 번호·[B#] 인용은 반드시 유지 (괄호 안이나 인용문
    안에 자연스럽게 삽입).
  - 주장에는 근거 (페이지·수식·실험 결과) 를 붙임. 근거 없으면 "확실치
    않습니다" 라고 솔직히.
  - "obvious", "trivially", "~인 것 같다" 같은 hedging 남용 금지.
    불확실할 땐 "제 생각엔 X 이고, 근거는 Y 인데, Z 때문에 확신은 낮습니다"
    같이 구체적으로.
- **겸양·사설 금지**: 미팅 톤이어도 "말씀드리자면", "사실" 같은
  filler 는 빼고 핵심부터.
- **표 사용 기준**: (a) 3 개 이상 row 의 병렬 비교, (b) identifiability
  matrix, (c) 수정사항 체크리스트. 그 외에는 본문에 녹여 씀.
- **예시 (권장)**:
  > "§2.2 의 Prop 9.5 를 다시 보니, 저자들의 증명 (pp. 7–9, eq. (9.12))
  > 에서 `‖K̃ − K‖₂ ≤ λ_{r+1}` 를 그대로 ρ² 로 나눠 최종 bound 를
  > `λ_{r+1}/(ρ² · √n)` 로 쓰고 있습니다. 여기서 ρ = 10⁻² 를 대입하면
  > 10⁴ factor 가 살아남는데, 논문의 실험 결과 (Fig. 7) 는 오차가
  > 실제로 그렇게 크지 않거든요. 제 생각엔 Caponnetto–De Vito 2007
  > [B16] 의 source condition 을 가정하면 `λ_{r+1}/ρ` 까지 줄일 수
  > 있을 것 같습니다. 문제는 source condition 이 eNTK spectrum 에서
  > 경험적으로 성립하는지인데, 이건 Lin 2025 [B28] 의 실측치를 보면..."
- **예시 (금지)**:
  > "- Prop 9.5 bound 는 too loose
  >  - ρ² blow-up
  >  - 개선안: d_eff
  >  - 조건: source condition"

  (단순 나열, 논리 연결 없음, 출처 없음.)

## Token efficiency rules (토큰 절약)
- Sub-agent 는 직전 sub-agent 가 이미 Read 한 파일을 통째로 재차 Read 하지 말 것.
  직전 산출물 (plan.md 등) 을 읽고 그 위에서 작업. references/ PDF 가
  반드시 필요하면 **특정 페이지만** pinpoint 하여 Read.
- WebSearch 는 직전 라운드에서 미처 다루지 못한 레퍼런스 확인 시에만.
  이미 bibliography 에 있는 URL 은 재검색 금지.
- 각 sub-agent 호출 시 메인 Claude 가 "이전 라운드 산출물 요약" 을 직접
  추출해 전달하지 말고, "state/iteration_NN/plan.md 읽고 작업" 식으로
  파일 경로만 지시. Sub-agent 가 필요한 만큼만 읽게 함.

## Math notation policy (PDF 렌더 제약)
- 산출물 PDF 는 pandoc + WeasyPrint 로 렌더되며, LaTeX 수식 매크로
  (`\operatorname`, `\hat`, `\frac`, `\sqrt`, `\sum`, `\int`, `\mathbb` 등)
  는 깨끗하게 렌더되지 않음. 따라서 **수식은 Unicode 문자로 직접** 작성.
- `$...$` 나 `$$...$$` LaTeX 블록 사용 금지.
- 권장 Unicode 표기:
  - 그리스: ρ, λ, μ, σ, θ, φ, ψ, Φ, Σ, Ω, ε, τ, ξ, ζ.
  - 연산·부등호: ≤, ≥, ≠, ≈, ∼, ±, ∞, →, ⇒, ∈, ∉, ⊆, ⊂, ∪, ∩.
  - 수학 집합: ℝ, ℕ, ℤ, ℚ, ℂ, 𝓗 (스크립트 H는 \U1D4D7 대신 "ℋ" 도 가능).
  - 함수·기호: √, Σ, ∏, ∫, ∂, ∇, ⊗, ⊕, ∘.
  - 첨자: 아래첨자는 `x_i` / `λ_{r+1}` 형식을 그대로 (가독성 우선), 또는
    유니코드 첨자 (x₁, λₙ, Hᵣ, A⁻¹, n², logₙ) 사용.
  - 햇·바: `φ̂` (U+0302), `x̄` (U+0304), `ỹ` (U+0303) 등 조합 diacritic.
- 분수는 `a/b` 또는 `(a) / (b)`; 복잡할 때 `√(λ_{r+1}/λ) · 1/√n` 같이 평문으로.
- 행렬·벡터는 `bold face` 대신 `**A**`, `**x**` (markdown bold) 또는
  그대로 `A`, `x` 로 충분. 필요 시 "matrix A" 같이 서술.
- 예시 (권장 ↔ 금지):
  - 권장: `Var(φ̂ᵢ) ≤ C · √(λ_{r+1}/λ) · 1/√n + O(log n / n²)`
  - 금지: `$\operatorname{Var}(\hat{\phi}_i) \le C\sqrt{\lambda_{r+1}/\lambda}\cdot 1/\sqrt{n}$`
- 원 논문 수식을 인용할 때도, 본문 안에서는 Unicode 로 재표기하고 원문은
  "(lrfshap.pdf, eq. (9))" 같이 **출처만** 명시.

<!-- ## 자원 상한
- 한 iteration 최대 12시간
- 한 실험 최대 6시간, 초과 시 자동 kill -->
## External resources policy
- Planner and Critic MUST actively use WebSearch and WebFetch to:
  - Find recent papers (arXiv, OpenReview, NeurIPS/ICLR/ICML proceedings, 2024-2026)
  - Read full PDF URLs when useful (WebFetch handles arXiv abs/pdf URLs)
  - Read relevant GitHub source files via https://raw.githubusercontent.com/... URLs
- When user places reference PDFs in `references/` folder, Planner and Critic 
  MUST read them with the Read tool (Claude Code's Read supports PDF).
  Treat `references/` as primary context before running web searches.
- When citing a source, include the URL or local path in the artifact.

## Reading large / figure-heavy PDFs (image interpretation capability)
- The Read tool refuses PDFs larger than ~20 MB and also returns text only
  for figure pages. Whenever a reference PDF is large OR contains important
  figures / plots / scanned tables, rasterize it first and Read the PNGs.
- Standard procedure (any sub-agent may invoke it via Bash):
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate freeshap
  python scripts/pdf_to_images.py <path/to.pdf>         # all pages, 150 DPI
  # or a subset:
  python scripts/pdf_to_images.py <path/to.pdf> --pages 6-12,21-26 --dpi 180
  ```
  Output goes to `<pdf_dir>/<pdf_stem>_img/page_NNN.png` plus a
  `pages.txt` text dump for cross-reference. Each PNG is well under the
  20 MB cap, so `Read` on a PNG returns the rendered page as an image that
  Claude can visually interpret (axis labels, curve shapes, captions, etc.).
- Planner / Critic / Executor MUST use this workflow whenever:
  (a) a reference PDF exceeds ~20 MB, or
  (b) the document's key evidence lives in figures (e.g. accuracy-vs-rank
      curves, TMC early-stop plots, Pareto frontiers).
  After rasterising, Read a targeted subset of pages — don't dump all pages
  into context. Start with pages containing Figures / Tables you actually
  need to reason about.
- For quick text-only extraction of a too-large PDF, pypdf (already in the
  `freeshap` env) is acceptable:
  ```python
  from pypdf import PdfReader; r = PdfReader(path); [p.extract_text() for p in r.pages]
  ```
  but do not rely on this alone when figures matter.
- Outputs under `references/<stem>_img/` and `references/<stem>_txt/` are
  cache artifacts — safe to delete and regenerate; do NOT commit them unless
  the user asks.

## Directive history (cumulative)
- Every user directive is stored as `state/directives/YYYYMMDD_iterNN.md`.
- `state/latest_directive.md` is a symlink to the most recent one.
- Agents should read `state/latest_directive.md` for the current task,
  but MAY read older files in `state/directives/` to understand the history.
- Never delete or rewrite past directive files.
