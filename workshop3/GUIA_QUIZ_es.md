# Guía de estudio para el quiz del Taller 3 (40 % de la nota)

El quiz evalúa si el equipo entiende **la formulación del MDP, el diseño del agente y los principios de RL**, y si sabe explicar **la función de recompensa, la estrategia de exploración y los resultados del entrenamiento**. Esta guía resume lo que hay que saber explicar con nuestras propias palabras. Los números son los de la corrida actual; si se vuelve a entrenar en Colab pueden cambiar un poco.

---

## 1. ¿Por qué RL y no supervisado?

- En los Talleres 1 y 2 el modelo **predice** si viene un fallo (dice *cuándo*), pero no dice **qué hacer**.
- Controlar el carro es una **secuencia de decisiones dependientes**: cada empujón cambia los estados futuros.
- No tenemos la "acción correcta" etiquetada para cada estado (el dataset tiene las acciones de un experto con ruido). Sí tenemos una **recompensa** (tiempo que el péndulo sigue arriba) y un **simulador** barato.
- El enunciado dice que RL no conviene si basta con supervisado; aquí la tarea es de control, así que RL encaja naturalmente.

## 2. El MDP (saberlo de memoria)

| Componente | Nuestro diseño | Por qué |
|---|---|---|
| Estado | `(x, x_dot, theta, theta_dot)`, dimensión 4 | Es **markoviano**: con esas 4 variables se conoce la evolución futura. Las 16 features del W1 son funciones del historial, no añaden información |
| Acciones | discretas: 0 = izquierda (−10 N), 1 = derecha (+10 N) | Igual que CartPole-v1; por eso DQN es adecuado |
| Transición | ecuaciones físicas del carro-péndulo + paso de Euler de 0.02 s | **Determinista** dada la acción; lo aleatorio es el estado inicial `U(−0.05, 0.05)` (y el ruido en la prueba con ruido) |
| Recompensa | +1 por paso sobrevivido (+ shaping opcional) | El retorno = pasos sobrevividos, máximo 500 |
| Terminación | fallo si \|θ\| > 12° o \|x\| > 2.4 m; **truncamiento** a 500 pasos | Los mismos límites que definían la etiqueta "peligro" del W1 |

**Pregunta típica: ¿diferencia entre terminación y truncamiento?** Terminar (fallo) significa que el futuro vale 0, así que en el objetivo TD se pone `(1 − d) = 0`. Truncar a 500 pasos es solo un límite de tiempo: el estado no es terminal, así que el agente **sí hace bootstrap**. Confundirlos enseña al agente que llegar a 500 es "malo".

## 3. La función de recompensa (lo más preguntado)

- **Recompensa de la tarea:** +1 por cada paso. Es densa en el tiempo, pero **no dice qué tan cerca está el fallo**; la única señal negativa es perder las recompensas futuras.
- **Shaping basado en potenciales:** `F = γ·Φ(s') − Φ(s)`, con `Φ(s) = −(k/2)·[(x/2.4)² + (θ/0.2095)²]`.
  - Viene del **W1**: la distancia a los límites fue lo que más predecía fallos (la regresión logística pasó de AUC 0.542 a 0.981 al añadirla).
  - **Teorema (Ng et al., 1999):** este tipo de shaping **no cambia la política óptima**. La suma de los `F` a lo largo de un episodio se cancela en cadena (telescópica) y queda ≈ `−Φ(s₀)`, una constante. Solo puede cambiar la **velocidad** de aprendizaje.
  - Por eso `Φ(terminal) = 0`.
- **Resultado:** con k = 1 el shaping **no tuvo efecto medible** (`no_shaping` dio casi lo mismo que `reference`). Tiene sentido: no puede cambiar *qué* aprende el agente.
- **Escasa vs densa (pregunta típica):** una recompensa escasa (solo al final) aprende lento; una densa mal diseñada puede llevar al agente a optimizar otra cosa. El shaping potencial es una forma *segura* de hacerla más densa.

## 4. El agente DQN

- Aproxima `Q(s, a)`: el retorno descontado esperado de tomar `a` en `s` y luego actuar de forma voraz.
- **Red:** MLP 4 → 128 → 128 → 2 (ReLU), unos 17 410 parámetros. Entradas divididas por escalas físicas fijas.
- **Objetivo TD:** `y = r + γ·(1 − d)·max_a' Q_target(s', a')`.
- Componentes que pide el enunciado y para qué sirve cada uno:
  - **Replay buffer (50 000):** muestrear al azar rompe la correlación entre pasos consecutivos.
  - **Red objetivo (copia cada 500 actualizaciones):** el objetivo no se mueve en cada paso, así el entrenamiento es más estable.
  - **ε-greedy con decaimiento:** explora al principio y explota al final.
  - **Gradient clipping (norma 10) + pérdida de Huber:** evitan actualizaciones gigantes cuando el error TD es grande.
- **Validación:** cada 10 episodios, el agente voraz (ε = 0) juega 10 episodios de validación; se guarda el checkpoint con mejor retorno de validación.

## 5. Exploración vs explotación

- ε empieza en 1 y se **multiplica por 0.99 al final de cada episodio** (mínimo 0.01). Baja de 0.1 tras unos 230 episodios.
- Con decaimiento **0.995**, ε sigue por encima de 0.2 hasta el final: más exploración, episodios de entrenamiento más cortos y ruidosos, y **no mejoró** el mejor retorno.
- Al final, con poca exploración, la política se degradó hacia **dejar que el carro se desplace** (fallo por posición): errores en estados poco visitados (los extremos del riel) que ya nadie corrige.

## 6. Resultados del entrenamiento (saber explicarlos)

- **Hay aprendizaje medible:** el retorno de validación sube de ~10 a varios cientos en pocos miles de pasos; `reference` llegó a 500 en las 3 semillas.
- **Patrón clave: todas las configuraciones aprenden y luego se degradan** (retorno final muy por debajo del mejor). Es una inestabilidad conocida de DQN (bootstrap con un objetivo que se mueve + aproximación de funciones).
- **Cómo lo mitigamos:** quedándonos con el **mejor checkpoint de validación**, no con el último.
- **Diagnóstico:** en los primeros ~2 mil pasos los episodios fallan porque cae el péndulo; cuando el agente ya sabe sostenerlo, casi todos los episodios que no llegan a 500 fallan porque el carro se sale del riel (también los de las políticas degradadas). Ese fallo llega ~100 pasos después, justo en el borde del horizonte de γ = 0.99 (1/(1−γ) ≈ 100).
- **γ = 0.95** llegó a 500 pero fue la que más se degradó (horizonte ≈ 20 pasos: "no ve" la deriva del carro).
- **lr = 1e-4:** ninguna semilla llegó a 500 en 300 episodios.

## 7. Prueba y comparación

- 100 estados iniciales **nunca usados** para entrenar ni para elegir; dos condiciones: nominal y con ruido de 2 N en la fuerza.
- **DQN:** ≈ 498 de 500 pasos, 97 % de éxito (nominal). Aleatoria ≈ 21.
- **Regla PD** (`θ + 0.5·θ̇ > 0` → derecha): resuelve todo (500). **El RL no la superó.** Es honesto decirlo: en un sistema simple y bien entendido, una regla de control es una línea base fuerte.
- **Clasificador W1 como política** (simula ambas acciones un paso adelante y elige la de menor probabilidad de peligro): ≈ 487. Ojo: usa un **modelo de la dinámica**, el DQN no.
- **Hallazgo interesante:** el DQN sobrevive **oscilando** el péndulo cerca del límite (±11°), mientras la regla PD lo mantiene casi vertical. El clasificador del W1 marca como peligrosos ~11 % de los pasos del DQN y ~0 % de la PD. Con ruido, el DQN cae a ≈ 371 (49 % de éxito). La recompensa +1 por paso **no premia el margen de seguridad**.
- Las otras dos semillas del DQN tenían tasa de peligro < 0.1 % y aguantaron bien el ruido: la tasa de peligro W1 detectó al agente frágil y la validación (500 en los tres) no.

## 8. Eficiencia de muestras

- El DQN llegó a un retorno suavizado de 195 tras ~6 mil pasos, y sus mejores checkpoints se guardaron tras ~16 mil pasos.
- El dataset del W1 tiene 35 874 pasos (24 782 filas etiquetadas de entrenamiento).
- **Matiz:** el supervisado aprende a *predecir*, no a *controlar*; para controlar necesitó un modelo de la dinámica. Y el dataset lo generó un experto, conocimiento que el DQN no tuvo.

## 9. Preguntas probables (respuestas cortas)

1. **¿Por qué DQN?** Acciones discretas y estado markoviano.
2. **¿Para qué la red objetivo?** Para que el objetivo TD no cambie en cada actualización.
3. **¿Para qué el replay buffer?** Para romper la correlación entre muestras consecutivas y reutilizar experiencia.
4. **¿Qué pasaría con γ = 0.95?** Horizonte más corto (≈ 20 pasos): no anticipa la deriva del carro y se degradó más.
5. **¿El shaping cambia la solución óptima?** No, es basado en potenciales; solo puede cambiar la velocidad de aprendizaje.
6. **¿Por qué el agente final no es el último?** Porque DQN se degrada; usamos el mejor checkpoint de validación.
7. **¿El RL superó al supervisado?** No lo superó en robustez; igualó al W1 guiado y quedó cerca de la regla PD en el caso nominal, sin modelo de la dinámica.
8. **¿Cómo mejorarían?** Validar también con ruido, recompensar el margen de seguridad (cambiando la tarea), Double DQN o actualizaciones suaves de la red objetivo.
