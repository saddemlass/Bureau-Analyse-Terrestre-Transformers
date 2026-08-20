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
| Lineaire | 0.5003 | 0.3204 | 7.16s |
| PyTorch | 0.5306 | 0.4322 | 52.95s |

Journal d'essais PyTorch : base: macro-F1=0.4322, temps=52.95s.

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
| emb_dim 48 | 28.54s | 1.86 | 0.4376 | +0.0054 |
| batch_size 512 | 41.19s | 1.29 | 0.4077 | -0.0245 |
| hidden_dim 80 | 52.02s | 1.02 | 0.4054 | -0.0268 |
| max_len 60 | 47.27s | 1.12 | 0.4322 | +0.0000 |
| patience 2 | 43.04s | 1.23 | 0.4322 | +0.0000 |

- temps Phase 3 : 52.95s
- macro-F1 Phase 3 : 0.4322
- temps Phase 5 : 32.85s
- macro-F1 Phase 5 : 0.4376
- facteur final : 1.61

Le modele final rapide combine les reglages retenus puis est reentraine proprement sur le meme split et les memes classes. Aller trop vite peut couter en generalisation : reduire trop le modele, tronquer trop le texte ou arreter trop tot peut enlever de l'information utile et degrade le macro-F1.

La comparaison temporelle est enregistree dans `outputs/phase5_time_comparison.png`. Les experiences sont enregistrees dans `outputs/phase5_experiments.csv`.

## Phase 6 — Le champ de vision du modèle

- longueur max avant troncature : 55 tokens
- longueur mediane : 14.0 tokens
- `max_len` accepte : 60 tokens, ce qui couvre 100.0 % des textes supervises
- architecture : `Embedding -> projection -> Conv1d dilatees residuelles + BatchNorm -> pooling masque -> MLP`

| couche | kernel | dilation | stride | champ ajoute | champ cumule |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 1 | 1 | 2 | 3 |
| 2 | 3 | 2 | 1 | 4 | 7 |
| 3 | 3 | 4 | 1 | 8 | 15 |
| 4 | 3 | 8 | 1 | 16 | 31 |
| 5 | 3 | 16 | 1 | 32 | 63 |

Le champ receptif final vaut 63 tokens, donc il couvre le `max_len` de 60. Avant entrainement, j'ai modifie le premier token d'un vrai releve : `saw` vers `zzztoken`. L'ecart max absolu des logits vaut 0.049449, donc la sortie change deja avec des poids identiques.

| modele | accuracy | macro-F1 | temps |
| --- | ---: | ---: | ---: |
| reference Phase 3/5 | 0.5283 | 0.4376 | 32.85s |
| convolution Phase 6 | 0.4495 | 0.1797 | 28.08s |

La courbe est `outputs/phase6_train_val_loss.png`, le tableau est `outputs/phase6_receptive_field.csv`.

## Phase 7 — Quatre relevés à la fois

Le modele Phase 6 contenait `BatchNorm1d`. Avec un batch de 4, ses statistiques de moyenne et variance dependent fortement des trois autres releves places dans le meme lot. Ces statistiques sont beaucoup plus bruitees qu'avec un batch de 128, et la prediction d'un releve depend alors du contenu des autres releves du batch.

Le diagnostic a aussi montre que l'ancien essai etait plafonne a seulement 10 lots par epoque : les gradients existaient, mais le modele voyait trop peu de donnees et finissait par predire presque uniquement `light`. La correction utilise `GroupNorm` dans les blocs Conv1d, donc la normalisation est calculee par releve et ne depend plus des autres exemples. Le batch utilise par le modele corrige reste bien 4.

| experience | accuracy | macro-F1 | temps |
| --- | ---: | ---: | ---: |
| ancien BatchNorm, batch=4 | 0.2443 | 0.0207 | 8.44s |
| corrige GroupNorm, batch=4 | 0.4697 | 0.2183 | 153.46s |
| corrige GroupNorm, batch normal | 0.4329 | 0.1650 | 29.81s |

Inference batch=1 : OUI. Figure : `outputs/phase7_batch4_comparison.png`.

## Phase 8 — Le Conseil a lu trois relevés

Liste interdite construite depuis les classes retenues et les fusions connues : changed, changing, changings, chevron, chevrons, cigar, cigars, circle, circles, cone, cones, cross, crosses, cylinder, cylinders, diamond, diamonds, disk, disks, egg, eggs, fireball, fireballs, flash, flashs, formation, formations, light, lights, oval, ovals, rectangle, rectangles, round, rounds, sphere, spheres, teardrop, teardrops, triangle, triangles.

Politique : je remplace les mots par `<MASKSHAPE>`. Le token garde l'information qu'un mot interdit etait present, mais interdit la recopie directe du nom de classe. Le remplacement utilise des bornes de mots, donc `light` ne coupe pas un mot plus long.

- releves avec mot interdit avant traitement : 44495
- releves avec mot interdit apres traitement : 0

| modele | accuracy | macro-F1 |
| --- | ---: | ---: |
| avant interdiction | 0.4697 | 0.2183 |
| apres interdiction | 0.3458 | 0.0944 |

Classes chutant le plus :

| classe | F1 avant | F1 apres | delta |
| --- | ---: | ---: | ---: |
| cigar | 0.5085 | 0.0000 | -0.5085 |
| sphere | 0.4374 | 0.0000 | -0.4374 |
| fireball | 0.6066 | 0.2145 | -0.3921 |

Comparaison demandee :

| classe | precision avant | recall avant | F1 avant | precision apres | recall apres | F1 apres |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| circle | 0.4188 | 0.1932 | 0.2644 | 0.3353 | 0.1364 | 0.1939 |
| light | 0.3828 | 0.9176 | 0.5402 | 0.3386 | 0.8874 | 0.4902 |
| triangle | 0.7046 | 0.6703 | 0.6870 | 0.5212 | 0.4827 | 0.5012 |

Le macro-F1 chute plus directement que l'accuracy quand les petites classes perdent leurs indices lexicaux. L'accuracy reste dominee par les classes frequentes, alors que le macro-F1 donne le meme poids moyen a chaque classe.

Fichiers : `outputs/phase8_class_scores.csv`, `outputs/phase8_before_after.png`.

## Phase 9 — Rendre des comptes sur trois décisions

Methode : occlusion leave-one-token-out sur la probabilite de la classe predite. Pour chaque token, je masque seulement ce token et je mesure la baisse de probabilite de la classe predite.

### correct
- index original : 69127
- datetime : 7/4/2011 13:00
- city : cooks mills (welland) (canada)
- vraie shape : disk
- shape predite : disk
- top2 : disk=0.998, circle=0.001
- marge top1-top2 : 0.9968
- temoignage : `White saucer appears over city and disappears`
- temoignage masque lu par le modele : `White saucer appears over city and disappears`
- mots les plus influents : saucer, and, white, appears, over
- Le modele retient surtout saucer, and, white, appears pour choisir `disk`.
- Il laisse peu peser disappears, city, over, alors qu'un humain lirait aussi le contexte complet.
- Ce cas montre que le dataset associe des mots descriptifs courts a `disk`/`disk`, pas une comprehension robuste de la scene.
- figure : `outputs\phase9_correct.png`
### wrong
- index original : 63313
- datetime : 7/15/2001 04:19
- city : bournemouth (uk/england)
- vraie shape : cross
- shape predite : light
- top2 : light=0.491, circle=0.100
- marge top1-top2 : 0.3905
- temoignage : `very bright static light sighted early morning in the East on July 15th 2001`
- temoignage masque lu par le modele : `very bright static  <MASKSHAPE>  sighted early morning in the East on July 15th 2001`
- mots les plus influents : bright, maskshape, east, 15th, in
- Le modele retient surtout bright, maskshape, east, 15th pour choisir `light`.
- Il laisse peu peser july, very, 2001, alors qu'un humain lirait aussi le contexte complet.
- Ce cas montre que le dataset associe des mots descriptifs courts a `cross`/`light`, pas une comprehension robuste de la scene.
- figure : `outputs\phase9_wrong.png`
### hesitant
- index original : 66041
- datetime : 7/24/2004 23:00
- city : lynnwood
- vraie shape : oval
- shape predite : light
- top2 : light=0.137, disk=0.137
- marge top1-top2 : 0.0000
- temoignage : `Near Seattle&#44 east to west siting&#44 full stop&#44 and then lit large object`
- temoignage masque lu par le modele : `Near Seattle&#44 east to west siting&#44 full stop&#44 and then lit large object`
- mots les plus influents : full, to, then, near, east
- Le modele retient surtout full, to, then, near pour choisir `light`.
- Il laisse peu peser object, lit, large, alors qu'un humain lirait aussi le contexte complet.
- Ce cas montre que le dataset associe des mots descriptifs courts a `oval`/`light`, pas une comprehension robuste de la scene.
- figure : `outputs\phase9_hesitant.png`

## Phase 10 — Chaque mot interroge les autres

Une tete d'attention manuelle projette chaque embedding en `Q`, `K` et `V`. `Q` represente la question posee par un token, `K` l'etiquette comparee chez les autres tokens, et `V` le contenu melange. Les scores sont calcules explicitement par `Q @ K.T / sqrt(d_k)`, puis `softmax` transforme chaque ligne en poids qui somment a 1. La sortie est le melange `weights @ V`.

- index original : 0
- datetime : 10/10/1949 20:30
- city : san marcos
- shape : cylinder
- commentaire : `This event took place in early fall around 1949-50. It occurred after a Boy Scout meeting in the Baptist Church. The Baptist Church sit`
- tokens : `['this', 'event', 'took', 'place', 'in', 'early', 'fall', 'around', '1949', '50', 'it', 'occurred', 'after', 'a', 'boy', 'scout', 'meeting', 'in', 'the', 'baptist', 'church', 'the', 'baptist', 'church', 'sit']`
- formes : X=(25, 32), Q=(25, 32), K=(25, 32), V=(25, 32), weights=(25, 25), output=(25, 32)
- preuve lignes = 1 : min=0.99999988, max=1.00000012, erreur max=1.1920929e-07
- figure : `outputs/phase10_attention_matrix.png`
- case pronom : ligne 0 (`this`), colonne 0 (`this`), poids=0.075994

Ces poids viennent d'un mecanisme non entraine : ils montrent comment lire la matrice, pas une comprehension de la coreference.

## Phase 11 — Le Conseil mélange vos mots

1. L'attention seule compare les contenus mais ne contient aucune information d'ordre.
2. Permuter les tokens permute les sorties de la meme facon : apres realignement, l'ecart est ~0.
3. L'encodage positionnel est ajoute aux embeddings avant `Q/K/V` ; la position influence donc questions, cles et valeurs.

- tokens originaux : `['this', 'event', 'took', 'place', 'in', 'early', 'fall', 'around', '1949', '50', 'it', 'occurred', 'after', 'a', 'boy', 'scout', 'meeting', 'in', 'the', 'baptist', 'church', 'the', 'baptist', 'church', 'sit']`
- tokens permutes : `['a', '50', 'in', 'in', 'it', 'boy', 'sit', 'the', 'the', '1949', 'fall', 'this', 'took', 'after', 'early', 'event', 'place', 'scout', 'around', 'church', 'church', 'baptist', 'baptist', 'meeting', 'occurred']`
- permutation : [13, 9, 4, 17, 10, 14, 24, 18, 21, 8, 6, 0, 2, 12, 5, 1, 3, 15, 7, 20, 23, 19, 22, 16, 11]
- ecart avant position : 8.940696716e-08
- ecart apres position : 0.127998054
- encodage : sinusoidal deterministe
- figure : `outputs/phase11_position_comparison.png`

## Phase 12 — Le prix des regards

Chaque token compare sa requete aux cles de tous les tokens. La matrice `weights` possede donc `n x n` coefficients : sa memoire croit en `O(n^2)`, et le produit `QK^T` augmente fortement avec la longueur. Le temps reel ne suit pas obligatoirement un facteur exact x4 a chaque doublement, car les petits tenseurs subissent l'overhead Python, le cache et la vectorisation.

| seq_len | temps median ms | cellules attention | memoire MiB |
|---:|---:|---:|---:|
| 16 | 0.115850 | 256 | 0.000977 |
| 32 | 0.089100 | 1024 | 0.003906 |
| 64 | 0.137600 | 4096 | 0.015625 |
| 128 | 0.162400 | 16384 | 0.062500 |
| 256 | 0.187000 | 65536 | 0.250000 |
| 512 | 0.470850 | 262144 | 1.000000 |

| doublement | ratio temps observe | ratio cellules |
|---:|---:|---:|
| 16 -> 32 | 0.769 | 4.0 |
| 32 -> 64 | 1.544 | 4.0 |
| 64 -> 128 | 1.180 | 4.0 |
| 128 -> 256 | 1.151 | 4.0 |
| 256 -> 512 | 2.518 | 4.0 |

Diagnostic log-log `log(time) ~ alpha * log(n)` : alpha=0.388. Figure : `outputs/phase12_attention_scaling.png`. Donnees : `outputs/phase12_attention_scaling.csv`.

## Phase 13 — Deux paires d'yeux

Une tete correspond a une famille de projections `Q/K/V`. Ici, deux tetes manuelles possedent deux familles independantes et travaillent chacune dans un sous-espace de dimension 16. Leurs sorties `[n,16]` sont concatenees en `[n,32]`, puis `Wo` les reprojette vers `d_model=32`.

- vrai releve : index 0, commentaire `This event took place in early fall around 1949-50. It occurred after a Boy Scout meeting in the Baptist Church. The Baptist Church sit`
- tokens : `['this', 'event', 'took', 'place', 'in', 'early', 'fall', 'around', '1949', '50', 'it', 'occurred', 'after', 'a', 'boy', 'scout', 'meeting', 'in', 'the', 'baptist', 'church', 'the', 'baptist', 'church', 'sit']`
- dimensions : X=(25, 32), head1 output=(25, 16), head2 output=(25, 16), concat=(25, 32), final=(25, 32)
- weights tete 1=(25, 25), weights tete 2=(25, 25)
- preuve lignes = 1 : erreur max tete 1=1.1920929e-07, erreur max tete 2=1.7881393e-07
- difference moyenne absolue weights1/weights2 : 0.01398874
- similarite cosinus aplatie : 0.90639651
- pronom : `this` ; tete 1 regarde surtout `50` poids=0.059527 ; tete 2 regarde surtout `1949` poids=0.074629
- figure : `outputs/phase13_two_heads.png`

Ces poids ne sont pas entraines : les deux projections produisent des patrons d'attention differents, mais on ne peut pas attribuer un role linguistique reel aux tetes.
