# Intent Qualification

Mădălin Ioana, 2026

## Problem

A trebuit sa construiesc un sistem care primeste un query in limbaj natural si returneaza companiile din dataset care se potrivesc cel mai bine cu cererea userului.

De exemplu, pentru query-ul "pharmaceutical companies in Switzerland with over 500 employees", sistemul ar putea returna:
- O firma de pharma din Basel cu 2000+ angajati, care este un match clar
- O firma elvetiana de biotech cu 600 angajati care face research, care este un caz discutabil
- O firma franceza de pharma cu departament R&D langa Geneva, care probabil nu ar trebui inclusa

Provocarea principala a fost sa diferentiez intre keyword match si intent match. Doua companii pot avea "logistics" in descriere, dar una face transport efectiv si alta vinde software. Similaritatea semantica nu e suficienta, a trebuit inteles ce face compania in practica si daca activitatea ei corespunde cu ce cauta utilizatorul.

## Approach

Am construit un pipeline cu 4 componente:

```
User Query → Intent Parser → Hard Filter → Embedding Ranker → LLM Classifier → Results
```

### Intent Parser

Prima etapa foloseste Llama-3.1-8b pentru a intelege ce vrea utilizatorul. Primeste query-ul si extrage:
- filters: constrangeri care se pot aplica direct pe coloane (is_public, min_employee_count, address, min_year_founded)
- semantic_intent: o descriere reformulata pentru embedding
- query_type: structured, semantic sau ecosystem, folosit mai tarziu pentru a decide cati candidati trimit la LLM

Am ales sa fac parsing cu LLM deoarece query-urile sunt foarte variate. "Public software companies with 1000+ employees" poate fi parsat cu regex, dar "B2B SaaS companies providing HR solutions" necesita intelegere semantica pentru a genera un semantic_intent potrivit.

### Hard Filter

Dupa parsing, aplic filtrele structurate direct pe dataframe.

Pentru geografie, campul address din dataset e un dict cu country_code (de exemplu 'ro', 'de', 'ch'). Nu se poate folosi str.contains("Romania") deoarece "Romania" nu apare nicaieri in date, doar codul ISO. Am construit o tabela de lookup care mapeaza nume de tari si regiuni la coduri ISO: "Romania" devine "ro", "Scandinavia" devine lista ["se", "no", "dk"], "Europe" devine lista completa de coduri europene.

Pentru date lipsa, am decis ca valorile NaN sa treaca prin filtru. Daca o companie nu are employee_count, nu o elimin atunci cand filtrez pentru "1000+ employees". Am facut aceasta alegere deoarece prefer sa am false positives pe care LLM-ul le poate corecta, decat false negatives pe care le pierd definitiv.

Daca hard filter-ul elimina tot, fac fallback la dataset-ul complet. Prefer un pool mai zgomotos decat zero rezultate.

### Embedding Ranker

Dupa hard filter raman intre 30 si 200 de companii. Le rankez folosind all-MiniLM-L6-v2 care ruleaza local pe CPU.

Pentru fiecare companie construiesc un string din nume, descriere, core_offerings, target_markets, business_model, NAICS label si adresa. Calculez embedding-ul pentru acest text si pentru semantic_intent, apoi sortez dupa cosine similarity.

Pastrez top 15-25 candidati in functie de query_type: 15 pentru structured deoarece filtrele au facut deja selectia principala, 20 pentru semantic si 25 pentru ecosystem deoarece acesta e cel mai greu de rankat corect.

Am ales all-MiniLM-L6-v2 deoarece e rapid (477 companii in 0.5 secunde) si nu necesita API extern. La 477 companii diferenta fata de modele mai mari e neglijabila deoarece LLM classifier-ul corecteaza oricum erorile de ranking.

### LLM Classifier

Ultimul pas este sa trimit batch-ul de candidati la Llama-3.3-70b. LLM-ul primeste query-ul original complet si pentru fiecare companie informatii despre nume, tara, descriere, NAICS, offerings, markets, business model, employee count, revenue, founding year si is_public.

Fiecare companie primeste un scor de la 0 la 100 si o explicatie scurta. Cele cu scor peste 60 sunt considerate calificate.

Prompt-ul are reguli stricte de evaluare, LLM-ul nu are voie sa presupuna ca o companie foloseste o tehnologie sau platforma doar pe baza industriei sau a marimii. Daca query-ul cere explicit "companies using Shopify" si nu exista dovezi in descrierea companiei, scorul trebuie sa fie sub 40. Astfel, sunt prevenite false positives bazate pe presupuneri.

Un aspect important, LLM-ul nu vede scorul de embedding. Judeca independent fiecare companie fata de query, ceea ce previne propagarea bias-ului din etapa anterioara.

Folosesc temperature=0 pentru rezultate deterministe. Acelasi query pe acelasi dataset produce rezultate identice, lucru important pentru debugging.

### De ce acest design

Am analizat alternativele inainte sa aleg aceasta arhitectura.

Prima varianta era sa trimit fiecare companie individual la LLM si sa intreb daca se potriveste cu query-ul. Ar fi insemnat 477 companii inmultit cu 12 queries, adica 5724 API calls. Prea scump, prea lent si rezultate inconsecvente intre rulari.

A doua varianta era sa folosesc doar embedding similarity. E rapid si ieftin, dar nu intelege intent-ul. Pentru query-uri despre furnizori de componente, embedding-ul tinde sa rankeze mai sus companiile care consuma acele componente decat pe cele care le produc deoarece consumatorii vorbesc mai mult despre ele in descrieri.

Pipeline-ul meu face 2 LLM calls per query (parse si classify) in loc de 477. Reduce costul de aproximativ 240 de ori mentinand acuratetea prin LLM final classification.

## Tradeoffs

Am optimizat pentru acuratete si recall, acceptand un cost LLM mai mare decat minimul posibil.

| Decizie | De ce | Cost |
|---------|-------|------|
| NaN passthrough in hard filter | Datele lipsa nu duc la descalificare | Ajung candidati irelevanti la LLM |
| all-MiniLM-L6-v2 local | Rapid, gratis si suficient pentru 477 companii | Intelegere semantica mai slaba decat modele mari |
| Single batch LLM call | Aproximativ 25x mai ieftin decat per companie | Context window limiteaza batch size |
| TOP_N intre 15 si 25 | Prinde cazuri la limita | Mai multi tokens per call |
| temperature=0 | Rezultate deterministe | Niciun dezavantaj pentru acest use case |

### Presupuneri

Am presupus ca NAICS codes sunt corecte. Ma bazez pe industry_naics ca ground truth pentru industrie, iar daca o companie pharma e clasificata gresit ca software, LLM-ul ar putea sa nu detecteze eroarea. In realitate unele companii au NAICS generic (541500, Computer Systems Design) care nu diferentiaza intre consultanta si product company.

Am presupus ca descrierea si detaliile de business sunt suficient de detaliate. LLM classifier-ul infereaza din aceste campuri, iar daca o companie are doar doua propozitii vagi la descriere, sistemul nu poate face reasoning complet.

Am presupus ca query-urile sunt in engleza. Intent parser-ul si semantic_intent sunt optimizate pentru engleza.

Am presupus ca query-urile cer liste de companii, nu o singura companie. Sistemul nu suporta explicit query-uri de tipul "cea mai mare companie din Romania" sau "top 3 firme de software". Trateaza aceste query-uri ca si cum ar cere toate companiile care se potrivesc, nu doar primele N. Pentru a imbunatati, as putea modifica IntentParser-ul sa extraga si chei de tipul {"sort_by": "revenue", "order": "desc", "limit": 1}

### Date lipsa

Sistemul e destul de robust, employee_count null trece filtrul numeric, revenue null trece filtrul numeric, is_public null trece filtrul boolean, description null inseamna embedding mai slab, dar NAICS si numele ajuta partial.

Aproximativ 30% din companii au employee_count null si 40% au revenue null. Decizia de NaN passthrough previne eliminarea lor prematura.

## Error Analysis

### Cazuri in care performeaza bine

Pe query-uri structurate cu geografie sistemul are rata de succes foarte mare: "Logistic companies in Romania", "Pharmaceutical companies in Switzerland", "Construction companies in US". Hard filter-ul reduce mult numarul de candidati, embedding-ul ii ordoneaza corect, iar LLM-ul face decizia finala.

La role disambiguation sistemul diferentiaza corect intre logistics operators (freight forwarding, transport) care sunt match, logistics software providers care nu sunt match si logistics consultants care nu sunt match. LLM classifier-ul cu business_model si NAICS face diferenta.

La regiuni geografice precum "Scandinavia", "Europe" sau "Balkans" mapping-ul pe coduri ISO prinde toate tarile din regiune.

### Limitari

La query-uri despre supply chain sau furnizori embedding-ul poate avea probleme. De exemplu, pentru un query despre furnizori de componente auto, embedding-ul poate ranka mai sus producatorii de masini decat producatorii de componente. Motivul e ca producatorii de masini vorbesc foarte mult in descrieri despre componentele pe care le folosesc ("advanced battery systems", "premium interior materials"), in timp ce furnizorii de componente au descrieri mai tehnice si mai putin apropiate semantic de produsul final. Am rezolvat partial prin formularea semantic_intent-ului sa descrie furnizorul, nu clientul si prin TOP_N mai mare (25) pentru ecosystem queries.

La query-uri vagi precum "fast-growing fintech" sistemul se bazeaza pe inferenta din founding year recent si limbaj in descriere, ceea ce e mai putin de incredere decat filtrarea structurata.

La companii cu date aproape lipsa, daca o companie care ar fi match perfect are description null, employee_count null si NAICS generic, sistemul o poate rata complet.

Un exemplu concret de misclassification ar fi pentru query-ul "automotive manufacturers founded before 2000" in care sistemul poate returna gresit suppliers de componente auto. Motivul e ca descrierile lor contin termeni precum "vehicle systems", "automotive engineering", "transportation solutions" care sunt semantic apropiate de manufacturing propriu-zis. Hard filter-ul le pastreaza (sunt fondate inainte de 2000), embedding-ul le pune sus (limbaj automotive), iar LLM-ul trebuie sa se bazeze pe NAICS (336300 vs 336100) si pe business_model pentru a le diferentia. Daca NAICS lipsesc sau business_model e vag, trec ca manufacturers.

### Semnalele pe care se bazeaza

Description e cel mai important semnal. LLM-ul infereaza aproape totul din descriere. Cand descrierea e detaliata sistemul functioneaza foarte bine. Cand e vaga, nu are de unde deduce ce face compania. Poate fi inselator cand o companie de software are descriere plina de termeni din industria pentru care face software, iar embedding-ul o rankeaza sus pentru query-uri despre acea industrie in loc de query-uri despre software.

NAICS codes sunt un discriminator puternic. NAICS 325412 (Pharmaceutical) descalifica automat o companie software, indiferent cum se descrie. Poate fi inselator cand codul e prea generic, 541500 acopera atat product companies cat si consulting.

Structured fields precum employee_count si revenue sunt exacte cand exista, dar lipsesc des si pot fi outdated. O companie marcata cu 150 angajati in 2020 poate avea 2000 acum.

## Scaling

### La 100000 companii

Pentru embedding as folosi un vector database precum FAISS sau Pinecone pentru approximate nearest neighbor search. Query time scade de la O(N) la O(log N). Embeddings-urile ar trebui calculate o singura data la ingestion si stocate, nu calculate la query time.

Pentru hard filter as adauga structured indexes: hash map pe country_code, array-uri sortate pentru employee_count si revenue (binary search), bitmap pentru is_public.

As adauga un cross-encoder reranker intre embedding si LLM. Cross-encoder e mai exact decat bi-encoder similarity, dar mult mai ieftin decat LLM. Ar reduce 100 candidati la 20 inainte de LLM classifier.

### La milioane de companii

La atat de multe companii problemele sunt diferite.

Nu mai merge pe un singur server, ai nevoie de vector DB distribuit si database sharded pe mai multe masini.

As face sharding geografic astfel incat "Companies in Romania" sa caute doar in shard-ul Europa, nu in toate milioanele de companii.

Embeddings pentru 10M companii ar dura aproximativ 160 minute pe CPU, deci ar fi nevoie de un batch job care ruleaza nightly pe GPUs.

La traffic mare costul LLM devine semnificativ. As folosi un model mai mic pentru parsing, caching agresiv pentru query patterns comune si mai multi provideri cu failover.

La 477 companii pot valida manual rezultatele. La milioane am nevoie de metrici calculate automat (NDCG, precision@k pe sample-uri). Fara monitorizare automata nu am cum sa stiu daca o schimbare imbunatateste sau degradeaza rezultatele.

## Failure Modes

### Erori cu scor mare

Daca NAICS lipsesc sau sunt gresite, o companie pharma cu o descriere apropiata de domeniul software ("digital platform", "cloud systems") poate trece ca software company cu scor mare.

Daca datele sunt outdated, o companie cu employee_count 150 din 2020, dar care acum are 2000, pica filtrul de "1000+ employees" desi ar fi match.

LLM-ul poate folosi world knowledge in loc de datele furnizate. Stie ca Novartis e pharma independent de ce scrie in dataset, ceea ce poate produce inconsecvente.

Pentru ecosystem queries embedding-ul poate ranka consumatorii mai sus decat producatorii. Daca toti producatorii reali sunt sub pozitia 25, LLM-ul nu ii vede niciodata.

### Comportament la erori API

Sistemul e fail-open. Daca intent parser-ul pica, query-ul original devine semantic_intent si se aplica zero filtre. Daca LLM classifier-ul pica, toti candidatii din embedding trec ca qualified. Am ales aceasta abordare deoarece prefer sa returnez rezultate imperfecte decat sa returnez eroare. In production as adauga alerting pentru aceste cazuri si retry logic.

### Monitorizare in production

Daca un filtru elimina peste 95% din dataset, e un semnal de alarma ca fie filtrul e prea strict, fie datele lipsesc pentru acel camp.

Daca toti candidatii au similarity sub 0.2, semantic_intent e slab formulat sau query-ul nu are match-uri bune in dataset.

Daca toti candidatii primesc scor 100 de la LLM, prompt-ul nu discrimineaza suficient. Daca toti primesc sub 50, posibil query-ul nu are match-uri reale.

Periodic as rula LLM-ul pe companii random din afara top 25 pentru a verifica false negatives. Daca gasesc companii cu scor mare acolo, embedding-ul rateaza ceva sistematic.

## Priority Improvements

Daca as continua dezvoltarea, in ordinea prioritatii:

* as adauga BM25 combinat cu embedding pentru a prinde exact keyword matches pe care embedding-ul le poate generaliza prea mult.

* as extrage explicit rolul de business (Supplier, Customer, Competitor) in parser pentru a clarifica directia la ecosystem queries.

* as adauga confidence scores per criteriu din LLM pentru debugging si explainability.

Pentru confidentialitatea datelor, Llama poate rula local in loc de Groq API. Datele de companii nu ar mai parasi infrastructura proprie.

## Conclusions

Pipeline-ul reduce costul de la aproximativ 2.8M tokens (varianta cu LLM per company) la aproximativ 120K tokens per run complet. E de 23 de ori mai ieftin mentinand acuratetea prin LLM final classification.

Sistemul functioneaza cel mai bine pe query-uri structurate cu geografie. Se confrunta cu dificultati la ecosystem queries si la companii cu date lipsa.

Alegerea tool-ului potrivit pentru fiecare etapa face diferenta, filtre pentru criterii structurate, embedding pentru retrieval semantic, LLM pentru clasificare finala. Combinatia produce rezultate mult mai bune decat oricare metoda folosita individual.
