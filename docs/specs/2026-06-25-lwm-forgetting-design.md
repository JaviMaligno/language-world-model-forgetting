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
- **Tamaño objetivo:** ~2–5k turnos.

### 5.2 Corpus general (replay para mixing)
- Slice de un corpus abierto general (FineWeb-Edu / Cosmopedia / wiki) como datos
  **single-turn** bajo el mismo objetivo de next-token. Se intercala con las trayectorias en
  el ratio correspondiente. Es la palanca anti-olvido clásica (rehearsal/replay).

## 6. Evaluación y métricas

Medir **antes** (modelo intacto) y **después** de cada corrida. Todo lo demás congelado
entre corridas de un mismo barrido.

| Dimensión | Instrumento | Métrica |
|---|---|---|
| Conocimiento | subset MMLU (~cientos Q), ARC-easy/challenge, factual QA | accuracy; Δ vs baseline |
| Modelado de lenguaje general | perplexity en texto general held-out | perplexity; Δ |
| Instruction-following (solo Instruct) | batería mini de formato/seguimiento | tasa de cumplimiento; Δ |
| Tarea aprendida (¿simula?) | trayectorias Terminal held-out | exact-match + fuzzy (token-F1 / edit normalizado) de la siguiente observación |

**Derivadas:**
- **Olvido** = caída en las métricas generales (knowledge + perplexity + IF).
- **Curva de retención** = métrica general vs mixing %.
- **Trade-off** = scatter (accuracy simulación) vs (Δ conocimiento).

**Avisos de interpretación (en el plan y el write-up):**
- No interpretar olvido si el modelo **no aprendió** la tarea (la comparación no significaría
  nada).
- Barrer ≥2 LR para no confundir un artefacto de un solo hiperparámetro con un hallazgo.

## 7. Infraestructura

- **Compute:** Azure `Standard_NC4as_T4_v3` **spot** en `australiaeast` (1× T4, 16 GB VRAM).
  Verificado 2026-06-25: subscription "Simple KYC Sandbox" tiene cuota NCASv3_T4 = 10 vCPUs
  y 50 vCPUs de low-priority en australiaeast (misma región que el Azure OpenAI existente).
- **Stack:** HuggingFace `transformers` + `trl` (SFT/CPT) + `peft` (LoRA). fp16 (T4 no tiene
  bf16). Gradient checkpointing + optimizador 8-bit (bitsandbytes) para que 1.5B full-FT
  entre en 16 GB.
- **Reproducibilidad:** seeds fijas; LR, pasos y eval set **congelados** entre corridas de
  mixing; versionar dataset, configs y resultados.
- **Repo:** `language-world-model-forgetting` (privado, espejo de `code-world-models`).
  Resultados a `docs/EXPERIMENTS.md`.

## 8. Modelos

- **Familia:** Qwen2.5 — `Qwen2.5-0.5B`, `Qwen2.5-0.5B-Instruct`, `Qwen2.5-1.5B`,
  `Qwen2.5-1.5B-Instruct`. Pesos abiertos, tamaños que caben en T4, family bien soportada.

## 9. Honestidad / límites (para el artículo)

- Escala de juguete ≠ prueba sobre los 35B/397B del paper. Estudiamos **la dinámica**, que
  aparece (más pronunciada) en modelos pequeños.
- LoRA enmascara el olvido por construcción → caveat explícito; por eso el estudio del olvido
  *crudo* es full-FT.
- El paper no mide la retención; nosotros no afirmamos que ellos tengan olvido, solo que la
  pregunta queda abierta y la cerramos a escala controlada.

## 10. Conexión con la serie del blog

Tercer artículo, espejo del de **deuda cognitiva** ("How Much Should You Still Know?") pero
del lado del *modelo*: ¿qué olvida un modelo cuando le enseñas a simular el mundo? Hermano
temático del experimento **Code World Models**. Eje compartido: **verificar sobre la señal
equivocada = falsa seguridad** (aquí: "no mediste el olvido → falsa sensación de que no lo
hay"; y "LoRA esconde el olvido").
