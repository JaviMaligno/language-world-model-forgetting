# El olvido del modelo del mundo
## Catastrophic forgetting en el continual pre-training de un Language World Model

**Fecha:** 2026-06-25
**Estado:** diseño aprobado, pendiente de plan de implementación
**Autor:** Javier Aguilar (con Claude)

---

## 1. Resumen ejecutivo

Reproducimos a escala de juguete la decisión de diseño central —y no medida— del paper
*Qwen-AgentWorld: Language World Models for General Agents* (Qwen/Alibaba, jun-2026,
arXiv:2606.24597): durante el **continual pre-training (CPT)** que convierte un LLM en
**modelo del mundo** (predecir `observación = f(historia, acción)`), ellos mezclan corpus
profesionales/generales y afirman que eso evita estrechar el modelo, **pero nunca miden
retención de conocimiento general** (no reportan MMLU ni equivalente antes/después).

Aislamos esa decisión: medimos cuánto conocimiento/capacidad general pierde un modelo
pequeño al hacer CPT sobre trayectorias de entorno estrechas, y trazamos la curva de
recuperación al subir el porcentaje de datos generales mezclados (*data mixing* = replay).

## 2. Pregunta de investigación e hipótesis

**RQ principal:** ¿El CPT sobre trayectorias de entorno estrechas degrada el conocimiento
general del modelo, y el data mixing lo mitiga? ¿A qué ratio?

**Sub-preguntas (ejes que exploramos según haya señal):**
- **H1 (curva mixing):** la pérdida de conocimiento general decrece monótonamente al subir
  el % de datos generales mezclados; existe un ratio "suficiente" con retorno decreciente.
- **H2 (LoRA esconde el olvido):** LoRA/QLoRA toca menos pesos → olvida menos *por
  construcción*, pero también aprende peor a simular el entorno. El método barato "oculta"
  el problema → falsa seguridad.
- **H3 (tamaño):** los modelos más pequeños (0.5B) olvidan más que los mayores (1.5B) bajo
  la misma receta.
- **H4 (base vs instruct):** el modelo Instruct se desmorona más (tenía más que perder:
  instruction-following + formato de chat) que el Base; el contraste es en sí un hallazgo.
- **H5 (trade-off):** al cruzar lo que se gana (simular el entorno) contra lo que se pierde
  (conocimiento general), aparece una frontera de Pareto; la pregunta honesta es si compensa.

**Por qué importa (gap):** el paper afirma-por-diseño que el mixing previene el
estrechamiento (lo enmarcan como *capacidad* —simular un hospital exige saber medicina—, no
como anti-olvido) pero no lo mide de frente. La retención es un beneficio que se solapa y
queda sin cuantificar. Lo cuantificamos.

> **Nota de honestidad sobre la cita:** arXiv:2606.24597 es de junio de 2026, posterior al
> corte de conocimiento del asistente. Antes de cualquier publicación hay que verificar la
> cita exacta (id, autores, claims citados textualmente). Independientemente del paper, el
> experimento se sostiene sobre la literatura clásica y citable de *continual learning*:
> catastrophic forgetting (McCloskey & Cohen 1989; French 1999), rehearsal/replay como
> mitigación, y la línea reciente de "fine-tuning degrada capacidades generales". El paper
> es la *motivación concreta*; la contribución es empírica y autónoma.

## 3. Variables independientes (la matriz)

| Eje | Valores |
|---|---|
| Mixing ratio | 0%, 10%, 25%, 50% de datos generales |
| Método | full fine-tuning, LoRA |
| Tamaño | Qwen2.5-0.5B, Qwen2.5-1.5B |
| Flavor | Base, Instruct |
| Control (no es eje) | barrido de LR (2–3 valores) — el olvido es muy LR-sensible |

El cruce completo (4×2×2×2 × LR × seeds) es inviable en una sola T4. Se ejecuta **por
fases**, dejando que la señal observada decida qué ejes merece profundizar.

**Eje v2 (opcional, según resultados):** *naturaleza del replay*. La v1 usa replay de
**texto plano** (preserva conocimiento, probablemente NO instruction-following). La v2
añade replay de **datos de instrucción** (formato chat) para testar si esa palanca recupera
el instruction-following que el texto plano no salva. Ver Fase 4 y §5.3.

## 4. Plan por fases

### Fase 0 — Harness + sanity (sin claims)
- Construir el dataset de Terminal (ver §5).
- Construir la batería de evaluación (ver §6) y medir los modelos **intactos** (baseline
  antes de tocar nada).
- Una corrida humo: CPT con 0% mixing en 0.5B. Objetivos: (a) confirmar que aprende a
  simular el entorno (accuracy held-out sube), (b) confirmar que sabemos medir el olvido,
  (c) fijar el LR y nº de pasos que producen olvido **visible pero no catastrófico** (un LR
  alto sobre datos estrechos da olvido dramático trivial; uno bajo, casi nada — hay que
  encontrar el régimen informativo).

### Fase 1 — La curva base (titular)
- Config: 0.5B-Instruct, full FT, mixing ∈ {0, 10, 25, 50}, 3 seeds por punto.
- Salida: **curva olvido-vs-mixing** con intervalos de confianza. Figura central del
  artículo.

### Fase 2 — Expandir ejes con señal
Solo los ejes que la Fase 1 sugiera jugosos:
- **Base vs Instruct** (H4): correr 0.5B-Base con la misma receta.
- **LoRA vs full** (H2): repetir la curva con LoRA/QLoRA.
- **Tamaño** (H3): 1.5B (base+instruct).

### Fase 3 — Trade-off (H5)
Scatter de **ganancia-en-entorno** (accuracy de simulación) vs **pérdida-general** (Δ
conocimiento) sobre todas las corridas → frontera de Pareto.

### Fase 4 — Naturaleza del replay (v2, opcional)
Solo si la Fase 1+ muestra que el modelo Instruct pierde instruction-following que el replay
de texto plano no recupera. Repetir la curva de mixing pero con el corpus de replay siendo
**datos de instrucción** (chat) en vez de texto plano. Pregunta: ¿qué palanca de replay
recupera qué capacidad? (texto → conocimiento; instrucción → IF). Comparar contra la v1.

## 5. Datos

### 5.1 Trayectorias de entorno (dominio Terminal)
- **Generación:** ejecutar comandos **reales** en un shell sandbox sobre un filesystem
  scratch efímero. Repertorio: operaciones de ficheros (`ls`, `cat`, `mkdir`, `rm`, `cp`,
  `mv`), `git`, snippets de `python`, `pip`, `grep`/`find`, etc. Capturar pares
  `(comando → stdout/stderr + exit code)`. Datos auténticos de world-modeling sin montar
  infra exótica.
- **Formato:** diálogo chat multi-turno — `system_prompt` + turnos donde `user` = acción
  (comando) y `assistant` = observación (salida). El system prompt declara que el assistant
  simula una terminal.
- **Expansión trayectoria-a-turno:** para una sesión de T turnos, cada turno t es un ejemplo
  de entrenamiento: input = historia (turnos 1..t-1) + comando t; target = salida t.
- **Loss masking:** la pérdida se computa **solo sobre tokens de observación**; los tokens
  de acción (`user`) se enmascaran (siguen en el forward como contexto, no contribuyen al
  gradiente). Réplica ligera del *information masking* del paper. (Opcional: filtrar turnos
  donde el estado no cambia para no enseñar a copiar.)
- **Tamaño:** lo que importa para el olvido son **pasos × LR sobre datos estrechos**, no el
  nº de turnos. Dimensionado: **~25–30k turnos ≈ ~8–10M tokens** (el generador los produce
  barato), split 90/10 train/held-out. Esto permite entrenar ~1–3 épocas sin caer en
  sobre-epocheo patológico (decenas de épocas memorizarían y contaminarían el resultado).

### 5.2 Corpus general (replay v1: texto plano)
- **Elegido: FineWeb-Edu.** Web filtrada por calidad educativa, moderno, rico en
  conocimiento, ampliamente usado y citable. Mejor proxy disponible de la distribución de
  pre-entrenamiento original (la mezcla exacta de Qwen no es pública). Datos **single-turn**
  bajo el mismo objetivo de next-token; se intercala con las trayectorias en el ratio
  correspondiente. Palanca anti-olvido clásica (rehearsal/replay). Se streamea de HF
  (~ilimitado; al 50% del presupuesto se necesitan ~5M tokens).
- **Banco de alternativas (no usadas de primeras, anotadas por si sirven):**
  - *Cosmopedia* — sintético tipo libro de texto; útil si queremos densidad de conocimiento
    estructurado, pero distribución sintética menos representativa.
  - *WikiText-103 / Wikipedia* — enciclopédico; OJO solapa con el *conocimiento* de
    MMLU/TriviaQA (no es train-on-test, pero ensucia la narrativa de "replay general").
  - *C4 / Dolma slice* — web genérica más amplia y ruidosa; alternativa si FineWeb-Edu
    resulta demasiado estrecha hacia lo educativo.

### 5.3 Corpus de instrucción (replay v2: chat) — opcional
- Para la Fase 4. Datos de instrucción abiertos en formato chat (p. ej. una mezcla tipo
  Tulu/OpenHermes/UltraChat — fijar el dataset exacto en el plan). Objetivo: testar si el
  replay de *instrucción* recupera el instruction-following que el texto plano no salva.

### 5.4 Presupuesto de entrenamiento (constante entre condiciones)
- **Clave del diseño limpio:** fijar el presupuesto de **tokens/pasos idéntico** en todas
  las celdas del barrido. El mixing **sustituye** fracción de tokens estrechos por generales
  (no añade encima) → el olvido es atribuible a la *distribución*, no a más cómputo.
- Punto de partida: ~1 época del dato estrecho al 0% ≈ ~8M tokens ≈ **~500 pasos** (seq
  1024, batch efectivo ~16). Barrer hasta 2–3 épocas. Congelado e idéntico salvo el eje que
  se estudia.

## 6. Evaluación y métricas

Medir **antes** (modelo intacto) y **después** de cada corrida. Todo lo demás congelado
entre corridas de un mismo barrido.

**Herramienta:** **EleutherAI lm-evaluation-harness** para TODO lo estándar (no subsets
caseros — añaden varianza y restan defensibilidad; el harness hace el loglikelihood scoring
correctamente, es el estándar citable, y en T4 con 0.5B los MC tardan minutos). Harness
custom **solo** para la tarea Terminal.

**Criterio de selección de instrumentos:** evitar el *suelo aleatorio*. En modelos 0.5–1.5B
muchos benchmarks ya están a random → sin rango para medir caída. Elegidos por tener el
baseline claramente por encima del suelo (con Qwen2.5, que es fuerte para su tamaño: 0.5B
≈45% MMLU, 1.5B ≈60%).

| Dimensión | Instrumento | Por qué / nota |
|---|---|---|
| **Canario continuo** | Perplexity en held-out general (WikiText-103 + slice C4) | Continuo, **sin suelo** → detector de drift más sensible |
| **Conocimiento factual** | TriviaQA closed-book (EM) | Mide *recall* de hechos = lo que se sobreescribe; el más directo para el titular. Generativo (más lento) |
| **Conocimiento académico** | MMLU (loglikelihood) | El benchmark que el paper omite; con Qwen hay recorrido; rápido |
| **Razonamiento/ciencia** | ARC-easy + ARC-challenge | Buen rango; estándar |
| **Sentido común** | HellaSwag + WinoGrande | Robustos, baseline muy sobre el suelo |
| **Instruction-following** (solo Instruct) | IFEval | Instrucciones verificables por programa → **determinista, sin LLM-juez** |
| **Tarea aprendida** (¿simula?) | Terminal held-out: EM + token-F1 / edit normalizado | Gate de validez: si no aprendió, el olvido no se interpreta |

**Derivadas:**
- **Olvido** = caída en las métricas generales (knowledge + perplexity + IF).
- **Curva de retención** = métrica general vs mixing %.
- **Trade-off** = scatter (accuracy simulación) vs (Δ conocimiento).

**Avisos de interpretación (en el plan y el write-up):**
- No interpretar olvido si el modelo **no aprendió** la tarea (la comparación no significaría
  nada).
- Barrer ≥2 LR para no confundir un artefacto de un solo hiperparámetro con un hallazgo.

## 7. Testing y validez experimental

Tres capas. (Lección de CWM: un bug de medición —timeout del sandbox → acuerdo 0 espurio—
casi produce un hallazgo falso. El análogo aquí, y el bug más peligroso, es el **loss
masking mal puesto**: enmascarar los tokens equivocados entrena otra cosa e invalida todo
sin avisar.)

**Capa A — Tests unitarios (código correcto):**
- **Máscara de pérdida (test #1, el crux):** el tensor de labels enmascara *exactamente* los
  tokens de acción/`user` y conserva los de observación.
- Expansión trayectoria-a-turno produce los `(input, target)` correctos; chat template y
  tokenización round-trip.
- El sampler de mixing produce el ratio pedido (ratio empírico sobre N batches ≈ objetivo).
- Generador Terminal: aislamiento del sandbox (no escapar del scratch dir), reproducibilidad.

**Capa B — Gates de validez experimental (el resultado *significa* algo):**
1. **Reproducción del baseline (gate maestro):** los números intactos de Qwen2.5 deben
   coincidir con el tech report de Qwen (±tolerancia). Si no, el harness está mal → valida
   todo el aparato de medición contra verdad publicada.
2. **Check de aprendizaje:** la accuracy de simulación held-out **debe subir** vs baseline; si
   no, el olvido no se interpreta.
3. **Overfit-one-batch:** un batch diminuto → loss→~0 confirma que el gradiente fluye a los
   tokens correctos.
4. **Control negativo:** 100% replay / 0% terminal → debe dar ~0 olvido y ~0 aprendizaje.
5. **Sin fuga train/test:** datos Terminal sintéticos disjuntos de los benchmarks; replay
   disjunto de los *items* de eval (spot-check).
6. **Determinismo:** misma seed → mismo resultado (re-correr una celda).

**Capa C — Estadística:** 3 seeds + intervalos de confianza en la curva titular; el barrido
de ≥2 LR es además test de robustez (que el hallazgo no sea artefacto de un hiperparámetro).

## 8. Infraestructura

- **Compute:** Azure `Standard_NC4as_T4_v3` **spot** en `australiaeast` (1× T4, 16 GB VRAM).
  Verificado 2026-06-25: subscription "Simple KYC Sandbox" tiene cuota NCASv3_T4 = 10 vCPUs
  y 50 vCPUs de low-priority en australiaeast (misma región que el Azure OpenAI existente).
- **Stack:** HuggingFace `transformers` + `trl` (SFT/CPT) + `peft` (LoRA) para entrenar;
  **`lm-evaluation-harness`** para los evals estándar. fp16 (T4 no tiene bf16). Gradient
  checkpointing + optimizador 8-bit (bitsandbytes) para que 1.5B full-FT entre en 16 GB.
  Collator custom con loss-masking sobre tokens de observación (el crux — ver §7).
- **Reproducibilidad:** seeds fijas; LR, pasos y eval set **congelados** entre corridas de
  mixing; versionar dataset, configs y resultados.
- **Repo:** `language-world-model-forgetting` (privado, espejo de `code-world-models`).
  Resultados a `docs/EXPERIMENTS.md`.

## 9. Modelos

- **Familia:** Qwen2.5 — `Qwen2.5-0.5B`, `Qwen2.5-0.5B-Instruct`, `Qwen2.5-1.5B`,
  `Qwen2.5-1.5B-Instruct`. Pesos abiertos, tamaños que caben en T4, family bien soportada.

## 10. Honestidad / límites (para el artículo)

- Escala de juguete ≠ prueba sobre los 35B/397B del paper. Estudiamos **la dinámica**, que
  aparece (más pronunciada) en modelos pequeños.
- LoRA enmascara el olvido por construcción → caveat explícito; por eso el estudio del olvido
  *crudo* es full-FT.
- El paper no mide la retención; nosotros no afirmamos que ellos tengan olvido, solo que la
  pregunta queda abierta y la cerramos a escala controlada.

## 11. Conexión con la serie del blog

Tercer artículo, espejo del de **deuda cognitiva** ("How Much Should You Still Know?") pero
del lado del *modelo*: ¿qué olvida un modelo cuando le enseñas a simular el mundo? Hermano
temático del experimento **Code World Models**. Eje compartido: **verificar sobre la señal
equivocada = falsa seguridad** (aquí: "no mediste el olvido → falsa sensación de que no lo
hay"; y "LoRA esconde el olvido").
