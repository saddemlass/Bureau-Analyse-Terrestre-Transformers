# Bureau d'Analyse Terrestre - La salle de lecture

## Phase 0 - Refaire les calculs du disparu

### Protocole

Le fichier source est telecharge automatiquement depuis la source officielle si `releves_klaxo3.csv` n'existe pas deja. Le dataset local n'est pas versionne.

Le chargement compte d'abord les lignes physiques du CSV. Ensuite, il charge les lignes a 11 champs, qui correspondent aux colonnes attendues. Les lignes avec trop de champs sont reparees quand le probleme vient de virgules non protegees dans `comments`. Les autres lignes malformees sont comptees a part. On ne les perd donc pas silencieusement.

Bilan du chargement :

- lignes physiques : 88875
- lignes chargees : 88875
- lignes reparees : 196
- lignes malformees non chargees : 0

### Choix de date

J'utilise `datetime` pour les calculs temporels. C'est la date de l'observation. `date_posted` est seulement la date de publication dans la base. Pour savoir quand les temoins disent avoir vu quelque chose dans le ciel, `datetime` est donc le bon champ.

Les heures `24:00` sont converties au jour suivant a `00:00`. Je ne supprime pas ces lignes, car elles portent une information temporelle recuperable.

### Periode testee

J'ai teste les donnees completes :

- releves : 88875
- date min : 1906-11-11
- date max : 2014-05-08
- jours calendaires couverts : 39261
- moyenne : 2.26 releves par jour

Les chiffres du dossier du predecesseur correspondent a la periode 1990-2014. C'est donc cette periode que j'utilise pour reproduire son dossier, sans forcer artificiellement les resultats.

Periode retenue :

- date min : 1990-01-01
- date max : 2014-05-08
- jours calendaires couverts : 8894
- releves retenus : 81678
- moyenne : 9.18 releves par jour

### Chiffres reproduits

- moyenne par jour : 9.18
- samedi : 17.7 %
- lundi : 12.6 %
- juillet : 11.3 %
- fevrier : 6.2 %

Pour le 4 juillet, le chiffre 51 ne correspond pas a un total et ne correspond pas a une journee particuliere. Il correspond a une moyenne calendaire arrondie : 1220 releves tombent un 4 juillet, sur 24 dates calendaires du 4 juillet couvertes par la periode. Cela donne 50.83, donc 51 releves arrondis.

La moyenne limitee aux 4 juillet qui ont au moins un releve est 53.04 releves, sur 23 dates observees. Le 4 juillet le plus charge est 2010-07-04, avec 206 releves.

### Chiffres supplementaires

- maximum quotidien toutes dates confondues : 206 releves
- date du maximum : 2010-07-04
- 4 juillet le plus charge : 2010-07-04, 206 releves
- rang du 4 juillet le plus charge parmi toutes les journees individuelles : 1

Ici, le rang du 4 juillet signifie : on classe chaque jour calendrier individuel par nombre de releves, puis on regarde la position du 4 juillet qui a le plus de releves.

### Top 10 des journees

| date | nombre_releves | jour_semaine | mois |
| --- | --- | --- | --- |
| 2010-07-04 | 206 | Sunday | 7 |
| 1999-11-16 | 195 | Tuesday | 11 |
| 2012-07-04 | 192 | Wednesday | 7 |
| 2013-07-04 | 180 | Thursday | 7 |
| 2011-07-04 | 155 | Monday | 7 |
| 2009-09-19 | 129 | Saturday | 9 |
| 2014-01-01 | 99 | Wednesday | 1 |
| 2013-12-31 | 96 | Tuesday | 12 |
| 2004-10-31 | 94 | Sunday | 10 |
| 2009-07-04 | 88 | Saturday | 7 |

Le tableau est aussi enregistre dans `outputs/phase0_top10_journees.csv`.

### Volume annuel

- 1990 : 295
- 1991 : 261
- 1992 : 291
- 1993 : 361
- 1994 : 476
- 1995 : 1441
- 1996 : 992
- 1997 : 1406
- 1998 : 1995
- 1999 : 3126
- 2000 : 3057
- 2001 : 3494
- 2002 : 3648
- 2003 : 4393
- 2004 : 4711
- 2005 : 4478
- 2006 : 4121
- 2007 : 4688
- 2008 : 5247
- 2009 : 4941
- 2010 : 4695
- 2011 : 5530
- 2012 : 7946
- 2013 : 7608
- 2014 : 2477

Verdict sur la phrase "le volume annuel croit continument jusqu'a la fin" : FAUX.

Des baisses existent : 1991 (295 vers 261), 1996 (1441 vers 992), 2000 (3126 vers 3057), 2005 (4711 vers 4478), 2006 (4478 vers 4121), 2009 (5247 vers 4941), 2010 (4941 vers 4695), 2013 (7946 vers 7608), 2014 (7608 vers 2477).

La courbe est enregistree dans `outputs/phase0_volume_annuel.png`.

## Phase 1 - Le chiffre etait vrai, la flotte est perdue

### Partie 1 - Ce que dit vraiment le chiffre du 4 juillet

Le chiffre du 4 juillet mesure un volume de signalements. Il dit qu'un certain jour, beaucoup de lignes ont ete enregistrees dans la base.

Il ne prouve pas a lui seul que les gens ne preteront pas attention a quelque chose. Il ne prouve pas non plus que les observations sont banales, que les temoins sont plus credules, ou que tous les evenements observes sont de meme nature.

Plusieurs explications peuvent produire le meme pic. Le 4 juillet, il y a des feux d'artifice. Plus de personnes sont dehors le soir. Il peut aussi y avoir plus de lumieres inhabituelles dans le ciel, ou plus de confusions avec des evenements lumineux connus. Les donnees permettent de constater le pic de signalements. Elles ne suffisent pas a expliquer toutes ses causes.

### Partie 2 - Trois releves reels

### Releve 1
- datetime : 2/9/1990 02:30
- lieu : plattsburg, ny, 
- shape : light
- comments : A bright yellow light follwing my car on the highway...and then just dissapeared.

### Releve 2
- datetime : 1/1/1990 18:00
- lieu : delhi (india), , 
- shape : triangle
- comments : Saw triangle shape plane with disco lights all over its edges standing over my house

### Releve 3
- datetime : 3/1/1990 21:00
- lieu : williston/ archer, fl, 
- shape : fireball
- comments : A blue&#44 firey ball&#44 about 20 feet wide and 150 feet in the air&#44 flies over head at a slow rate making no sound and lighting up the goun

Ces trois lignes sont comptees de la meme maniere dans les statistiques. Pourtant, elles ne racontent pas la meme chose. Une ligne peut etre courte, une autre peut decrire une lumiere, une autre peut raconter une observation plus precise ou plus confuse. Le comptage mesure d'abord un volume de temoignages.

### Partie 3 - Nouvelle tache ML

ENTREE : le texte `comments` ecrit par le temoin.

SORTIE : la forme `shape` observee.

"A partir du temoignage ecrit par un temoin, predire la forme de l'objet qu'il decrit."

## Phase 2 — Test d'acceptation : mémoriser 8 relevés

Cette phase fait volontairement de l'overfit. Le but est de verifier que toute la chaine fonctionne sur un cas minuscule : texte, representation numerique, tenseur PyTorch, labels, loss, optimisation et prediction.

Ce test ne mesure absolument pas la generalisation. Il ne dit pas si le modele saura reconnaitre correctement de nouveaux temoignages. Il dit seulement qu'un petit reseau peut memoriser huit exemples reels quand toute la plomberie d'apprentissage est coherente.

Les huit releves sont choisis de maniere deterministe dans le dataset deja charge, avec `comments` non vide et `shape` non vide. J'utilise une tokenisation simple en lowercase, puis un bag-of-words de comptage. Le vocabulaire est construit uniquement sur ces huit commentaires. Il contient 98 mots.

Les classes presentes sont : `circle`, `disk`, `fireball`, `light`, `other`, `sphere`, `triangle`, `unknown`.

Le modele PyTorch est volontairement petit :

- `Linear(input_dim, 16)`
- `ReLU`
- `Linear(16, n_classes)`

L'optimizer est Adam, avec un learning rate de 0.05. La loss est `CrossEntropyLoss`.

Le modele atteint 8/8 en 13 iterations. La loss passe de 2.1124 au depart a 0.0049 a la fin. La courbe de loss est enregistree dans `outputs/phase2_overfit_loss.png`.

Conclusion : si un petit reseau ne peut pas memoriser huit exemples, il existe probablement un probleme dans la representation, les labels, la loss, l'optimisation ou la chaine d'entrainement.

## Phase 3 - Battre le service statistique

Regle finale `shape` : les lignes sans `shape` restent dans le dataset general mais sont exclues de cette tache supervisee. Notre dataset charge compte 3118 `shape` manquantes : 2922 dans les lignes CSV directement valides, plus 196 dans les lignes malformees reparees. `unknown` et `other` sont retires car ce sont des categories fourre-tout. Les doublons evidents sont fusionnes : `round` vers `circle`, `changed` vers `changing`. Les 6 classes ayant moins de 10 exemples apres nettoyage sont exclues, car elles ne representent que 14 releves et sont trop petites pour etre reparties proprement entre train, validation et test. Aucun sous-echantillonnage artificiel par classe n'est applique.

- lignes initiales : 88875
- shape manquantes : 3118
- unknown : 6319
- other : 6247
- lignes/classes avant filtre <10 : 73183 / 25
- classes <10 retirees : {'delta': 8, 'crescent': 2, 'pyramid': 1, 'flare': 1, 'hexagon': 1, 'dome': 1}
- lignes retirees car classe trop rare : 14
- lignes gardees : 73169
- nombre final de classes : 19
- classes finales : changing, chevron, cigar, circle, cone, cross, cylinder, diamond, disk, egg, fireball, flash, formation, light, oval, rectangle, sphere, teardrop, triangle
- split : train 51218, validation 10975, test 10976

Le vocabulaire du reseau est construit uniquement sur train. Exemple de passage numerique :

- texte brut : `saw a formation of three lights hovering over a cliff with a beam from the largest light`
- tokens : `['saw', 'a', 'formation', 'of', 'three', 'lights', 'hovering', 'over', 'a', 'cliff', 'with', 'a', 'beam', 'from', 'the', 'largest']`
- ids : `[29, 5, 68, 7, 74, 9, 54, 14, 5, 3624, 19, 5, 463, 26, 3, 4259]`

Modele PyTorch : `Embedding` avec mean pooling masque, puis `Linear`, `ReLU`, `Dropout`, `Linear`. Pas de RNN, pas d'attention, pas de Transformer. La metrique principale est le macro-F1.

| modele | accuracy test | macro-F1 test | temps entrainement |
| --- | ---: | ---: | ---: |
| Majorite | 0.2443 | 0.0207 | 0.00s |
| Lineaire | 0.5003 | 0.3204 | 5.78s |
| PyTorch | 0.5306 | 0.4322 | 51.48s |

Journal d'essais PyTorch : base: macro-F1=0.4322, temps=51.48s.

La courbe train loss / validation loss est enregistree dans `outputs/phase3_train_val_loss.png`.

## Phase 4 - Le carnet de pannes

| panne | geste exact | signature observee | test diagnostic rapide | correction |
| --- | --- | --- | --- | --- |
| 1 | Evaluer avec `model.train()` donc dropout actif | score test instable : 0.4151 a 0.4248, alors que le train est 0.0485 | repeter deux fois la meme evaluation sans changer les donnees | appeler `model.eval()` avant validation/test |
| 2 | Decaler volontairement le mapping de sortie au decodage | la loss validation descend mais le macro-F1 interprete tombe a 0.0116 | comparer `class_to_id` et `id_to_class` utilises au train et au reporting | conserver un mapping unique et versionne pendant toute l'experience |
| 3 | Mettre un learning rate quasi nul (`1e-7`) | loss train quasi plate : 2.9189 vers 2.9173 | verifier la norme des gradients et la valeur du learning rate | remettre un learning rate compatible avec Adam (`0.003` ici) |

Figures : `outputs/phase4_panne1.png`, `outputs/phase4_panne2.png`, `outputs/phase4_panne3.png`.

## Phase 5 - Le budget de calcul

| reglage | temps | facteur gain | macro-F1 | ecart de score |
| --- | ---: | ---: | ---: | ---: |
| emb_dim 48 | 29.54s | 1.74 | 0.4376 | +0.0054 |
| batch_size 512 | 41.91s | 1.23 | 0.4077 | -0.0245 |
| hidden_dim 80 | 131.40s | 0.39 | 0.4054 | -0.0268 |
| max_len 60 | 49.84s | 1.03 | 0.4322 | +0.0000 |
| patience 2 | 37.72s | 1.36 | 0.4322 | +0.0000 |

- temps Phase 3 : 51.48s
- macro-F1 Phase 3 : 0.4322
- temps Phase 5 : 28.17s
- macro-F1 Phase 5 : 0.4376
- facteur final : 1.83

Le modele final rapide combine les reglages retenus puis est reentraine proprement sur le meme split et les memes classes. Aller trop vite peut couter en generalisation : reduire trop le modele, tronquer trop le texte ou arreter trop tot peut enlever de l'information utile et degrade le macro-F1.

La comparaison temporelle est enregistree dans `outputs/phase5_time_comparison.png`. Les experiences sont enregistrees dans `outputs/phase5_experiments.csv`.
