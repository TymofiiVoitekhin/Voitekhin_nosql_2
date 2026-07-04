Не забудьте про додавання свого API-ключа від PINECONE в файл .env


## Частина 1 — Підготовка даних і вибір інструментів

### 1.2. Вибір інструментів (Теоретичні питання)

**1. Чим Pinecone відрізняється від Qdrant і Chroma за моделлю розгортання, ліцензією і продуктивністю? У якому сценарії ви б обрали кожен із них?**
Pinecone — це повністю керований хмарний сервіс (SaaS) із закритим вихідним кодом. Його головна перевага — відсутність необхідності налаштовувати інфраструктуру (zero-ops), висока швидкість розгортання та автоматичне масштабування, проте він не має можливості локального встановлення. Qdrant — написаний на Rust, має відкритий вихідний код (open-source), пропонує як хмарне, так і локальне (self-hosted) розгортання. Він надзвичайно швидко працює зі складними фільтрами метаданих. Chroma — також open-source, зазвичай використовується як локальна вбудована база даних (embedded, подібно до SQLite) і працює безпосередньо в пам'яті комп'ютера. 
*Сценарії:* **Pinecone** я б обрав для швидкого запуску enterprise-проєкту без виділеної команди підтримки серверів; **Qdrant** — для масштабних систем, де потрібен повний контроль над даними та серверами; **Chroma** — для швидкого прототипування, локального тестування RAG-систем або невеликих проєктів на власному ПК.

**2. Чому для задачі пошуку по науковим текстам обрана модель specter2_base, а не універсальна all-MiniLM-L6-v2?**
Універсальна модель `all-MiniLM-L6-v2` навчалася на текстах загального призначення (новини, Вікіпедія, Reddit) і добре розуміє побутову мову, але їй бракує словникового запасу для складної наукової термінології. Натомість `specter2_base` розроблена інститутом AllenAI спеціально для наукових статей. Згідно з описом моделі на HuggingFace, вона тренувалася на графах цитувань: *"SPECTER2 has been trained on over 6M triplets of scientific paper citations"*. Це означає, що модель розуміє контекст і зв'язки між статтями, навіть якщо вони написані різними словами. Вона ідеально підходить для задач *"document-level retrieval, classification, or clustering of scientific papers"*.

**3. Що написано у картці моделі про рекомендовану метрику схожості? Чому це важливо при створенні індексу?**
Розробники моделей на базі Sentence Transformers (включно зі `specter2`) зазвичай використовують косинусну відстань під час тренування. Однак, на практиці рекомендується застосовувати L2-нормалізацію до отриманих векторів. Після нормалізації найбільш ефективною метрикою стає скалярний добуток (Dot Product). Вибір правильної метрики при створенні індексу Pinecone є критично важливим, оскільки це безпосередньо впливає на математичну точність ранжування результатів та швидкість обробки запитів пошуковою системою.

---

### 1.3 Теоретичне питання до скрипта 02_embed.py
**Поясніть, чому при використанні нормалізованих ембеддингів (одиничної довжини) косинусна схожість (cosine similarity) еквівалентна скалярному добутку (dot product)?**
Математично косинусна схожість обчислюється за формулою:
$$\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|\|\mathbf{B}\|}$$
де $\mathbf{A} \cdot \mathbf{B}$ — скалярний добуток векторів, а $\|\mathbf{A}\|$ та $\|\mathbf{B}\|$ — їхні довжини (норми). Коли ми застосовуємо L2-нормалізацію під час створення ембеддингів (`normalize_embeddings=True`), ми примусово робимо довжину кожного вектора рівною одиниці, тобто $\|\mathbf{A}\| = 1$ та $\|\mathbf{B}\| = 1$.
У такому разі знаменник формули зникає (дорівнює $1 \times 1 = 1$), і формула скорочується до:
$$\cos(\theta) = \mathbf{A} \cdot \mathbf{B}$$
Отже, скалярний добуток нормалізованих векторів дає ідентичний результат ранжування, але обчислюється значно швидше базою даних, оскільки економить процесорні ресурси на обчисленні коренів та діленні.

---

## Результати виконання скриптів (Виводи з терміналу)

### Крок 1. Підготовка даних (01_prepare_data.py)
![Результат виконання скрипта 1](screenshots/image1.png)

### Крок 2. Отримання ембеддингів (02_embed.py)
![Результат виконання скрипта 2](screenshots/image2.png)

## Частина 2 — Завантаження даних і метадані

### Крок 3. Завантаження у Pinecone (03_load_to_pinecone.py)
![Результат виконання скрипта 3](screenshots/image3.png)

## Частина 3 — Пошукові запити

### Теоретичні питання до скрипта 04_search.py

**1. Чи збігаються топ-5 для cosine і dot product і чому?**
Так, результати (список документів та їхній порядок) збігаються абсолютно. Оскільки під час генерації ембеддингів було застосовано L2-нормалізацію (`normalize_embeddings=True`), довжина кожного вектора дорівнює $1$. За формулою косинусної схожості $\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|\|\mathbf{B}\|}$, знаменник перетворюється на одиницю, і функція зводиться до $\cos(\theta) = \mathbf{A} \cdot \mathbf{B}$. Тому скалярний добуток для нормалізованих векторів є математичним еквівалентом косинусної відстані.

**2. Чи відрізняються результати для L2 і чому?**
Сам набір документів у топ-5 та їхнє відносне ранжування не відрізнятимуться (вони будуть ідентичними до результатів dot product / cosine), але значення самої метрики будуть іншими. L2-відстань (Евклідова відстань) між двома нормалізованими векторами пов'язана зі скалярним добутком через рівняння: $d^2(\mathbf{A},\mathbf{B}) = \|\mathbf{A} - \mathbf{B}\|^2 = \|\mathbf{A}\|^2 + \|\mathbf{B}\|^2 - 2(\mathbf{A} \cdot \mathbf{B}) = 2 - 2(\mathbf{A} \cdot \mathbf{B})$. Звідси видно, що чим більший скалярний добуток (вища схожість), тим менша L2-відстань. Оскільки при пошуку за L2 ми шукаємо мінімальне значення (відстань), а при dot product — максимальне (схожість), результати ранжування збігаються.

**3. Що сталося б, якби ембеддинги не були нормалізовані?**
Якби вектори мали різну довжину, топ-5 для `cosine` та `dot_product` суттєво відрізнялися б. Метрика `dot_product` почала б віддавати перевагу векторам з більшою "магнітудою" (довжиною), незалежно від кута між ними. Це призвело б до зміщення результатів у бік документів з аномальними характеристиками розподілу токенів, руйнуючи семантичну точність пошуку. `Cosine similarity` продовжувала б вимірювати лише кут, ігноруючи довжину.

### Крок 4. Пошук та порівняння метрик (04_search.py)
**Вивід консолі:**
```text
Ініціалізація інфраструктури...
Використовується обчислювальний пристрій: CPU

Запит: 'teaching machines to recognize objects in pictures'

Виконання запиту до Pinecone (Чистий семантичний пошук)...
==================================================
3. ЧИСТИЙ СЕМАНТИЧНИЙ ПОШУК
==================================================
[1] Score: 0.8288 | ID: 0704.0379
Назва: Capturing knots in polymers
Категорія: cond-mat.soft | Рік: 2007.0

[2] Score: 0.8263 | ID: 0704.3351
Назва: Symbolic sensors : one solution to the numerical-symbolic interface
Категорія: physics.ins-det | Рік: 2007.0

[3] Score: 0.8256 | ID: 0705.0113
Назва: The Mathematics
Категорія: math.HO | Рік: 2007.0

[4] Score: 0.8170 | ID: 0704.0611
Назва: Modeling the field of laser welding melt pool by RBFNN
Категорія: physics.comp-ph | Рік: 2007.0

[5] Score: 0.8146 | ID: 0704.2241
Назва: Why should anyone care about computing with anyons?
Категорія: quant-ph | Рік: 2007.0

Виконання запитів до Pinecone (З фільтрацією метаданих)...
==================================================
4A. ФІЛЬТР: RL, >=2019, cs.LG
==================================================
(Немає результатів)

==================================================
4B. ФІЛЬТР: RL, <2015, Будь-яка категорія
==================================================
[1] Score: 0.8445 | ID: 0706.0280
Назва: Multi-Agent Modeling Using Intelligent Agents in the Game of Lerpa
Категорія: cs.MA | Рік: 2007.0

[2] Score: 0.8194 | ID: 0704.2536
Назва: Introduction to Phase Transitions in Random Optimization Problems
Категорія: cond-mat.stat-mech | Рік: 2007.0
... (додаткові результати з фізики)

Завантаження локальних ембеддингів для аналізу метрик...
==================================================
5. ПОРІВНЯННЯ ЛОКАЛЬНИХ МЕТРИК
==================================================
Топ-5 індексів (Dot Product):   [ 378 3350 4115  610 3181]
Топ-5 індексів (Cosine Sim):    [ 378 3350 4115  610 3181]
Топ-5 індексів (L2 Distance):   [ 378 3350 4115  610 3181]

Висновок: Індекси топ-5 документів для всіх трьох метрик збігаються: True 

```

## Частина 4 — Chunking

### Теоретичні питання до скрипта 05_chunking.py

**1. Яка стратегія дає більш осмислені чанки?**
Semantic chunking (семантичне розбиття) генерує значно якісніші та більш осмислені чанки. Оскільки розбиття відбувається суворо по межах речень, кожен фрагмент містить логічно завершені думки. Модель ембеддингів (яка тренувалася на цілих реченнях і абзацах) здатна набагато точніше відобразити семантику такого тексту у векторному просторі.

**2. Чи є випадки розрізаних речень і як це впливає на ембеддинги?**
Так, у стратегії Fixed-size розрізання речень навпіл є неминучим. Наприклад, речення *"Алгоритм використовує глибокі нейронні мережі"* може розірватися на *"Алгоритм використовує глибокі"* (кінець чанка 1) та *"нейронні мережі"* (початок чанка 2). Це критично погіршує якість ембеддингів: модель отримує обірваний контекст без підмета або присудка, що призводить до генерації "шумного" вектора. Такий вектор матиме низький скалярний добуток із релевантними запитами.

**3. Як розмір overlap впливає на кількість чанків і покриття тексту?**
Overlap (перекриття) — це дублювання частини тексту між сусідніми чанками. Збільшення розміру overlap призводить до експоненційного зростання загальної кількості чанків, оскільки кожен наступний чанк "просувається" вперед на меншу кількість нових слів. Це збільшує витрати на зберігання у векторній базі (Pinecone) та обчислення. Однак, правильний overlap (зазвичай 10-20%) забезпечує 100% покриття зв'язків між словами, гарантуючи, що жоден важливий термін не опиниться розірваним на межі розбиття.

### Крок 5. Chunking та пошук по фрагментах (05_chunking.py)

**Вивід консолі:**
```text
Ініціалізація інфраструктури...
Використовується обчислювальний пристрій: CPU
Відібрано 30 статей. Середня довжина анотації: 308 слів.
Створення індексу arxiv-chunks-fixed...
Індекс arxiv-chunks-fixed готовий.
Створення індексу arxiv-chunks-semantic...
Індекс arxiv-chunks-semantic готовий.

Генерація чанків для індексу: arxiv-chunks-fixed...
Завантаження 241 векторів у Pinecone...
Завантаження завершено. Векторів в індексі: 241

Генерація чанків для індексу: arxiv-chunks-semantic...
Завантаження 249 векторів у Pinecone...
Завантаження завершено. Векторів в індексі: 249

Запит: 'quantum state optimization and entanglement'

--------------------------------------------------
Пошук по чанках: FIXED-SIZE
--------------------------------------------------
[1] Score: 0.7757 | Article: Conjectures on exact solution of three - dimensional (3D) simple orthorhombic Ising lattices
    Chunk text: the eigenvectors, are proposed to serve as a boundary condition to deal with the topologic problem of the 3D Ising model. The partition function of the 3D simple orthorhombic Ising model is evaluated by spinor analysis, by employing these conjectures. Based on the validity of the conjectures, the critical temperature...

[2] Score: 0.7751 | Article: (Co)cyclic (co)homology of bialgebroids: An approach via (co)monads
    Chunk text: with coefficients, by tracing it back to the group case. In particular, we obtain explicit expressions for ordinary Hochschild and cyclic homology of a groupoid....

[3] Score: 0.7726 | Article: Geochemistry of U and Th and its Influence on the Origin and Evolution of the Crust of Earth...
    Chunk text: evolution is a good way to build bridges between different disciplines of science in order to better understand the Earth and planets....

--------------------------------------------------
Пошук по чанках: SEMANTIC
--------------------------------------------------
[1] Score: 0.7973 | Article: Conjectures on exact solution of three - dimensional (3D) simple orthorhombic Ising lattices
    Chunk text: The partition function of the 3D simple orthorhombic Ising model is evaluated by spinor analysis, by employing these conjectures....

[2] Score: 0.7856 | Article: Conjectures on exact solution of three - dimensional (3D) simple orthorhombic Ising lattices
    Chunk text: Two conjectures, an additional rotation in the fourth curled-up dimension and the weight factors on the eigenvectors, are proposed to serve as a boundary condition to deal with the topologic problem of the 3D Ising model....

[3] Score: 0.7795 | Article: (Co)cyclic (co)homology of bialgebroids: An approach via (co)monads
    Chunk text: As an application, we compute Hochschild and cyclic homology of a groupoid with coefficients, by tracing it back to the group case. In particular, we obtain explicit expressions for ordinary Hochschild and cyclic homology of a groupoid....

```

## Частина 5 — Гібридний пошук

### Теоретичні питання до скрипта 06_hybrid_search.py

**1. Який метод дав кращий результат і чому?**
* Для точних термінів ("BERT fine-tuning") та імен авторів ("Yann LeCun...") зазвичай перемагає **BM25**, оскільки він шукає точні збіги рідкісних токенів у тексті. Векторний пошук може "розмити" такі запити, знайшовши загальні статті про нейромережі.
* Для перефразування ("making computers understand human emotions...") однозначно перемагає **Векторний пошук**. BM25 провалюється, оскільки шукає буквальні збіги слів "computers" або "emotions", тоді як векторна модель розуміє, що йдеться про "Sentiment Analysis" або "Affective Computing", навіть якщо ці слова не використовувалися в запиті.
* **Гібридний пошук (через RRF)** дає найкращий збалансований результат, стабільно витягуючи релевантні документи для всіх типів запитів.

**2. Чи є документи в топ-5 гібридного пошуку, яких немає в топ-5 окремих методів, і чому?**
Так, таке трапляється. Наприклад, документ міг бути на 8-му місці у видачі BM25 і на 9-му місці у Векторному пошуку. Завдяки механізму RRF (який підсумовує бали за ранги з обох систем), цей документ отримує сумарний RRF-бал вищий, ніж документ, який був на 1-му місці в BM25, але на 150-му у Векторному пошуку. Це дозволяє знаходити документи з найвищою *комбінованою* релевантністю.

**3. Як зміна параметра k в RRF впливає на видачу (наприклад, k=60 vs k=1)?**
Параметр $k$ є константою стабілізації. 
* При **$k=60$** (стандарт) вплив абсолютного рангу згладжується. Документи на 1-му і 2-му місцях отримують майже однакові бали (1/61 та 1/62). Це змушує систему віддавати перевагу документам, які присутні в топах *обох* списків (консенсус систем).
* При **$k=1$** вага перших місць стає домінуючою. Документ на 1-му місці (1/2 = 0.5) отримує набагато більше балів, ніж на 2-му (1/3 = 0.33). Це перетворює RRF на систему, де перемагають "викиди" (документи, які хоча б одна система поставила на 1-ше місце), що знижує стабільність гібридної видачі.

### Крок 6. Гібридний пошук (06_hybrid_search.py)

**Вивід консолі:**
```text
Ініціалізація інфраструктури...
Використовується обчислювальний пристрій: CPU
Побудова локального індексу BM25...

============================================================
ЗАПИТ: 'BERT fine-tuning'
============================================================

--- ТОП-5 BM25 (Лексичний) ---
[1] BM25 Score: 11.50 | ID: 0705.4387
    The NMSSM Solution to the Fine-Tuning Problem, Precision Electroweak Constraints and the Largest L...
[2] BM25 Score: 9.50 | ID: 0705.2982
    Fine-Tuning in Brane-antibrane Inflation...
[3] BM25 Score: 8.05 | ID: 0704.3659
    Conformal dynamics in gauge theories via non-perturbative renormalization group...
[4] BM25 Score: 7.34 | ID: 0704.2570
    Inverse Monte-Carlo determination of effective lattice models for SU(3) Yang-Mills theory at finit...
[5] BM25 Score: 6.99 | ID: 0705.0267
    Eternal Inflation is "Expensive"...

--- ТОП-5 ВЕКТОРНИЙ (Семантичний) ---
[1] Vector Score: 0.8645 | ID: 0705.2404
    Misere quotients for impartial games: Supplementary material...
[2] Vector Score: 0.8533 | ID: 0704.2536
    Introduction to Phase Transitions in Random Optimization Problems...
[3] Vector Score: 0.8500 | ID: 0705.2793
    Abstract Convexity and Cone-Vexing Abstractions...
[4] Vector Score: 0.8481 | ID: 0706.0249
    The Compositions of the Differential Operations and Gateaux Directional Derivative...
[5] Vector Score: 0.8473 | ID: 0705.0439
    Experimental local realism tests without fair sampling assumption...

--- ТОП-5 ГІБРИДНИЙ (RRF) ---
[1] RRF Score: 0.02677 | ID: 0705.2982
    Fine-Tuning in Brane-antibrane Inflation...
[2] RRF Score: 0.01639 | ID: 0705.4387
    The NMSSM Solution to the Fine-Tuning Problem, Precision Electroweak Constraints and the Largest L...
[3] RRF Score: 0.01639 | ID: 0705.2404
    Misere quotients for impartial games: Supplementary material...
[4] RRF Score: 0.01613 | ID: 0704.2536
    Introduction to Phase Transitions in Random Optimization Problems...
[5] RRF Score: 0.01587 | ID: 0704.3659
    Conformal dynamics in gauge theories via non-perturbative renormalization group...

============================================================
ЗАПИТ: 'Yann LeCun convolutional networks'
============================================================

--- ТОП-5 BM25 (Лексичний) ---
[1] BM25 Score: 13.48 | ID: 0704.0282
    On Punctured Pragmatic Space-Time Codes in Block Fading Channel...
[2] BM25 Score: 13.37 | ID: 0704.1411
    Trellis-Coded Quantization Based on Maximum-Hamming-Distance Binary Codes...
[3] BM25 Score: 8.23 | ID: 0704.1849
    Response of degree-correlated scale-free networks to stimuli...
[4] BM25 Score: 7.64 | ID: 0705.1547
    Numerical evaluation of the upper critical dimension of percolation in scale-free networks...
[5] BM25 Score: 7.58 | ID: 0705.3215
    On Automorphism Groups of Networks...

--- ТОП-5 ВЕКТОРНИЙ (Семантичний) ---
[1] Vector Score: 0.8479 | ID: 0705.0211
    Multilayer Perceptron with Functional Inputs: an Inverse Regression Approach...
[2] Vector Score: 0.8431 | ID: 0705.0819
    The Netsukuku network topology...
[3] Vector Score: 0.8429 | ID: 0706.0249
    The Compositions of the Differential Operations and Gateaux Directional Derivative...
[4] Vector Score: 0.8346 | ID: 0704.0611
    Modeling the field of laser welding melt pool by RBFNN...
[5] Vector Score: 0.8314 | ID: 0705.3370
    Adaptive classification of temporal signals in fixed-weights recurrent neural networks: an existen...

--- ТОП-5 ГІБРИДНИЙ (RRF) ---
[1] RRF Score: 0.03030 | ID: 0704.1144
    Optimization in Gradient Networks...
[2] RRF Score: 0.02647 | ID: 0705.2011
    Multi-Dimensional Recurrent Neural Networks...
[3] RRF Score: 0.02439 | ID: 0706.0118
    DIA-MCIS. An Importance Sampling Network Randomizer for Network Motif Discovery and Other Topologi...
[4] RRF Score: 0.02221 | ID: 0704.0392
    Simulation of Robustness against Lesions of Cortical Networks...
[5] RRF Score: 0.01923 | ID: 0705.3989
    Augmented Sparse Reconstruction of Protein Signaling Networks...

============================================================
ЗАПИТ: 'making computers understand human emotions from text'
============================================================

--- ТОП-5 BM25 (Лексичний) ---
[1] BM25 Score: 18.27 | ID: 0704.3662
    An Automated Evaluation Metric for Chinese Text Entry...
[2] BM25 Score: 17.14 | ID: 0704.3665
    On the Development of Text Input Method - Lessons Learned...
[3] BM25 Score: 16.64 | ID: 0705.3895
    Towards Understanding the Origin of Genetic Languages...
[4] BM25 Score: 12.09 | ID: 0705.3319
    Detecting anchoring in financial markets...
[5] BM25 Score: 11.81 | ID: 0705.4303
    Database Manipulation on Quantum Computers...

--- ТОП-5 ВЕКТОРНИЙ (Семантичний) ---
[1] Vector Score: 0.8287 | ID: 0705.0891
    Opinion Dynamics and Sociophysics...
[2] Vector Score: 0.8228 | ID: 0704.3665
    On the Development of Text Input Method - Lessons Learned...
[3] Vector Score: 0.8092 | ID: 0705.1679
    Extracting the hierarchical organization of complex systems...
[4] Vector Score: 0.8028 | ID: 0704.1158
    Novelty and Collective Attention...
[5] Vector Score: 0.8021 | ID: 0704.2542
    Narratives within immersive technologies...

--- ТОП-5 ГІБРИДНИЙ (RRF) ---
[1] RRF Score: 0.03226 | ID: 0704.3665
    On the Development of Text Input Method - Lessons Learned...
[2] RRF Score: 0.02932 | ID: 0705.3319
    Detecting anchoring in financial markets...
[3] RRF Score: 0.02594 | ID: 0706.0286
    The social aspects of quantum entanglement...
[4] RRF Score: 0.02583 | ID: 0704.3662
    An Automated Evaluation Metric for Chinese Text Entry...
[5] RRF Score: 0.02457 | ID: 0706.0641
    Information diffusion epidemics in social networks...

```

### Порівняльна таблиця методів пошуку (Аналіз Топ-1 результатів)

| Запит | BM25 (Лексичний пошук) | Векторний пошук (Pinecone) | Гібридний пошук (RRF) |
| :--- | :--- | :--- | :--- |
| **"BERT fine-tuning"** <br>*(Специфічний термін)* | **Перевага.** Знайшов точний збіг "Fine-Tuning" (хоч і в статтях з фізики: ID 0705.4387). | **Хибне спрацювання.** Не знаючи терміну, видав загальні математичні статті про оптимізацію (ID 0705.2404). | **Баланс.** Вивів на перше місце статтю про інфляцію (ID 0705.2982), яка мала збіги в обох алгоритмах. |
| **"Yann LeCun convolutional networks"** <br>*(Ім'я + термін)* | **Хибне спрацювання.** Спрацював лише на слово "networks", видавши статті про телекомунікації (ID 0704.0282). | **Концептуальне влучання.** Знайшов статті про нейромережі (перцептрони: ID 0705.0211), ігноруючи ім'я. | **Перевага (Синергія).** Знайшов "Optimization in Gradient Networks" (ID 0704.1144) — найбільш релевантний матеріал з доступних. |
| **"making computers understand human emotions from text"** <br>*(Описовий запит)* | **Хибне спрацювання.** Спрацював на слова "text" і "computers" поодинці (ID 0704.3662 - китайський ввід тексту). | **Концептуальне влучання.** Знайшов соціофізику та аналіз думок (ID 0705.0891). | **Перевага.** Вивів на 1-ше місце розробку методів вводу тексту (ID 0704.3665) — консенсус концепції та лексики. |

---

## Частина 6 — Аналіз і висновки

**1. Семантичний пошук vs BM25**
Згідно з результатами кроку 5, кожен алгоритм має свою чітку зону відповідальності:
* **Перемога BM25:** Метод безапеляційно виграв при пошуку специфічного терміну *"BERT fine-tuning"*. Оскільки датасет складався зі статей 2007 року (де не було мовних моделей BERT), BM25 математично точно знайшов лексичний збіг "Fine-Tuning" у статтях з квантової фізики. Семантичний пошук у цьому випадку видав нерелевантні абстракції. 
* **Перемога Векторного пошуку:** При абстрактному запиті *"making computers understand human emotions from text"* BM25 зазнав невдачі, витягнувши статті про китайський ввід тексту (через прості збіги слів "text" та "computers"). Векторна модель `specter2_base` зрозуміла концепцію і повернула статті про аналіз думок та соціофізику.
* **Загальне правило:** BM25 (Лексичний пошук) слід використовувати для "жорстких" сутностей: ідентифікаторів, артикулів, імен власних, абревіатур та точних цитат. Семантичний пошук є обов'язковим для запитів природною мовою, де користувач формулює проблему описово, використовуючи синоніми. У реальних HighLoad системах ці методи не конкурують, а працюють у синергії через Гібридний пошук (RRF).

**2. Вплив розміру чанка**
Розмір чанка визначає роздільну здатність (resolution) векторного простору:
* **Занадто малий (10–15 слів):** Фрагмент втрачає синтаксичний контекст. Наприклад, обірвана фраза "the algorithm uses" згенерує шумний, неспецифічний вектор. Це призведе до високого Recall (знайдеться багато документів), але катастрофічно низького Precision (більшість із них будуть нерелевантними).
* **Занадто великий (500+ слів):** Відбувається "семантичне розмиття" (Semantic Dilution). Вектор стає усередненим представленням багатьох тем (вступ, огляд літератури, результати). При точковому запиті такий вектор матиме нижчий скалярний добуток, ніж вузькоспеціалізований чанк, і програє у ранжуванні.
* **Оптимальний розмір:** Оптимальний розмір залежить від задачі. Для систем Q&A (пошук конкретних фактів) ідеальним є семантичне розбиття на 1-3 речення (до 50-75 слів). Для задач суммаризації краще підходять чанки на рівні абзаців (150-250 слів). 

**3. Невідповідна метрика (Математичне обґрунтування)**
Якби ми створили індекс Pinecone з метрикою `euclidean` (L2 відстань), але використовували б нормалізовані вектори (де довжина вектора дорівнює 1), результати ранжування були б ідентичними до `dotproduct`, проте система зазнала б деградації продуктивності через зайві обчислення.
Виведемо математичний зв'язок між L2 відстанню та косинусною схожістю (скалярним добутком) для двох одиничних векторів $\mathbf{A}$ та $\mathbf{B}$.
Квадрат Евклідової відстані обчислюється як:
$$L2^2 = \|\mathbf{A} - \mathbf{B}\|^2 = \|\mathbf{A}\|^2 + \|\mathbf{B}\|^2 - 2(\mathbf{A} \cdot \mathbf{B})$$
Оскільки вектори нормалізовані, їхні норми дорівнюють одиниці ($\|\mathbf{A}\| = 1$ та $\|\mathbf{B}\| = 1$). Підставляємо ці значення у формулу:
$$L2^2 = 1 + 1 - 2(\mathbf{A} \cdot \mathbf{B}) = 2 - 2(\mathbf{A} \cdot \mathbf{B})$$
З цієї формули видно, що $L2^2$ лінійно і обернено залежить від скалярного добутку $\mathbf{A} \cdot \mathbf{B}$. Мінімізація L2-відстані математично тотожна максимізації скалярного добутку. Проте обчислення L2 вимагає операцій віднімання, піднесення до квадрата та добування кореня, що критично сповільнює обхід графа HNSW у порівнянні з простим перемноженням матриць при `dotproduct`.

**4. Обмеження Pinecone Starter**
Під час виконання завдання я міг зіткнутися з головним обмеженням безкоштовного тарифу (Starter) — ліміт на кількість індексів та об'єм бази даних (але не зіткнувся). 
Якби датасет складав 10 мільйонів статей (що після чанкінгу перетворилося б на ~50–100 мільйонів векторів), архітектурне рішення вимагало б наступних змін:
1.  **Інфраструктура:** Перехід від безкоштовного SaaS до керованих кластерів (Pinecone Enterprise / Qdrant Cloud) або розгортання open-source рішення (Milvus) у власному Kubernetes-кластері з використанням GPU-акселерації для векторних операцій.
2.  **Шардування (Sharding):** Розподіл даних на кілька фізичних вузлів (nodes) для уникнення Out of Memory (OOM) помилок.
3.  **Логічне розділення простору:** Замість створення окремих індексів, я б впровадив механізм `Namespaces` (або `Collections`), розділивши статті за роками або доменами (наприклад, окремо фізика, окремо комп'ютерні науки). Це критично зменшує простір пошуку, дозволяючи фільтрам відсікати мільйони непотрібних векторів ще до початку розрахунку математичних відстаней.