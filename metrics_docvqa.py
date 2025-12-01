import math

# ------------------------
# 1. Levenshtein distance
# ------------------------
def levenshtein_distance(s: str, t: str) -> int:
    s = s or ""
    t = t or ""
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n
    
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s[i-1] == t[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,     # borrado
                dp[i][j-1] + 1,     # inserción
                dp[i-1][j-1] + cost # sustitución
            )
    return dp[n][m]

# ------------------------
# 2. Normalized Lev. Sim
# ------------------------
def normalized_lev_sim(pred: str, gt: str) -> float:
    pred = (pred or "").strip().lower()
    gt   = (gt or "").strip().lower()
    
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    
    d = levenshtein_distance(pred, gt)
    return 1.0 - d / max(len(pred), len(gt))

# ------------------------
# 3. ANLS para un ejemplo
# ------------------------
def anls_single(pred: str, gt: str, tau: float = 0.5) -> float:
    """
    ANLS = max(0, 1 - dist/len) si la similitud >= tau, si no = 0.
    tau ~ 0.5 como en benchmarks DocVQA.
    """
    sim = normalized_lev_sim(pred, gt)
    return sim if sim >= tau else 0.0

# ------------------------
# 4. Retrieval Precision@k
# ------------------------
def retrieval_precision_at_k(gt_page: int, pred_pages: list[int], k: int) -> float:
    """
    gt_page: índice de página ground truth (0-based)
    pred_pages: lista de índices de página de los chunks recuperados (0-based)
    k: número de chunks considerados
    """
    if not pred_pages:
        return 0.0
    k = min(k, len(pred_pages))
    for j in range(k):
        if pred_pages[j] == gt_page:
            return 1.0
    return 0.0

# ------------------------
# 5. sim(c_j, a) para ChunkScore
# ------------------------
def substring_max_sim(chunk: str, answer: str) -> float:
    """
    sim(cj, a): máxima similitud (normalizada) entre la respuesta
    y cualquier subcadena de igual longitud dentro del chunk.
    """
    chunk = (chunk or "").lower()
    answer = (answer or "").lower()
    L = len(answer)
    if L == 0 or len(chunk) == 0 or len(chunk) < L:
        return 0.0
    
    best = 0.0
    for i in range(0, len(chunk) - L + 1):
        sub = chunk[i:i+L]
        sim = normalized_lev_sim(sub, answer)
        if sim > best:
            best = sim
    return best

# ------------------------
# 6. Chunk Score@k (eq. 4)
# ------------------------
def chunk_score_at_k(retrieved_chunks: list[str], answer: str, k: int) -> float:
    """
    Chunk Score@k = log2(1 + max_{1<=j<=k} sim(cj, a))
    Según ecuación (4) del paper. :contentReference[oaicite:3]{index=3}
    """
    if not retrieved_chunks:
        return 0.0
    k = min(k, len(retrieved_chunks))
    sims = [substring_max_sim(retrieved_chunks[j], answer) for j in range(k)]
    max_sim = max(sims) if sims else 0.0
    return math.log2(1.0 + max_sim)
