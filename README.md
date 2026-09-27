# chessengine

Un moteur d'échecs écrit en Rust, avec une interface web locale pour jouer contre lui et le regarder apprendre.

- **Partie** : joue contre le moteur (clic ou glisser-déposer), avec sa réflexion en direct : évaluation, profondeur, positions par seconde et ligne principale.
- **Entraînement** : le moteur joue contre lui-même, ajuste son évaluation, puis la nouvelle version affronte l'ancienne. Graphiques de progression, partie en direct, valeur des pièces et cartes de chaleur qui évoluent pendant l'apprentissage.

## Lancer le projet (Mac)

1. Installer Rust (une seule fois) :
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
2. Récupérer le code et lancer :
   ```bash
   git clone https://github.com/HarpeLm/chessengine.git
   cd chessengine
   cargo run --release
   ```
   Le navigateur s'ouvre sur <http://localhost:8080>. La première compilation prend une minute.

Toujours utiliser `--release` : le moteur est environ 20 fois plus rapide qu'en mode debug.

Autres commandes :

| Commande | Rôle |
| --- | --- |
| `cargo run --release -- --port 8081` | interface sur un autre port |
| `cargo run --release -- uci` | mode UCI, pour brancher le moteur sur Cute Chess, Arena ou lichess-bot |
| `cargo run --release -- perft 6` | compte les positions à la profondeur 6 (doit donner 119 060 324) |
| `cargo run --release -- bench` | vitesse de recherche du champion enregistré |
| `cargo test --release` | lance les tests |

Les poids appris et l'historique d'entraînement sont enregistrés dans le dossier `data/`. Supprime-le pour tout remettre à zéro.

## Comment le code est organisé

| Fichier | Contenu |
| --- | --- |
| `src/board.rs` | l'échiquier : pièces, cases, lecture et écriture FEN, jouer un coup, cases attaquées, hachage Zobrist |
| `src/movegen.rs` | génération des coups légaux, notations UCI (`e2e4`) et SAN (`Nf3`) |
| `src/perft.rs` | le test perft, qui prouve que la génération des coups est exacte |
| `src/eval.rs` | l'évaluation classique : matériel, tables de position, pions passés, mobilité… |
| `src/nnue.rs` | le réseau de neurones d'évaluation (NNUE) : calcul rapide, mise à jour incrémentale, entraînement |
| `src/search.rs` | la recherche : alpha-bêta, approfondissement itératif, quiescence, coup nul, réductions, tri des coups |
| `src/tt.rs` | la table de transposition (le cache des positions déjà analysées) |
| `src/train.rs` | l'apprentissage par parties contre soi-même (méthode de Texel) |
| `src/game.rs` | la partie jouée dans le navigateur |
| `src/server.rs`, `src/app.rs` | le serveur web local et l'état partagé |
| `src/uci.rs` | le protocole UCI |
| `web/` | l'interface (HTML, CSS, JavaScript sans framework) |

Ordre conseillé pour lire le code : `board.rs` → `movegen.rs` → `perft.rs` → `eval.rs` → `search.rs` → `train.rs`.

## Comment le moteur apprend

L'évaluation est une somme pondérée d'environ 400 critères. Chaque critère a deux poids : un pour le milieu de partie, un pour la finale. Une génération d'entraînement se déroule en quatre étapes :

1. **Parties** : le champion joue contre lui-même depuis des ouvertures tirées au hasard. Chaque position calme est notée avec le résultat final de la partie.
2. **Ajustement** : une descente de gradient cherche les poids dont l'évaluation prédit le mieux ces résultats. Un rappel vers les poids actuels évite qu'ils partent n'importe où quand les données manquent.
3. **Match** : le candidat affronte le champion. Chaque ouverture est jouée deux fois, en inversant les couleurs. S'il marque plus de 50 %, il passe à la vérification.
4. **Vérification** : un second match, sur d'autres ouvertures. Le candidat ne devient champion que s'il gagne aussi ce match, et c'est lui qui mesure le gain d'ELO. Le premier match a servi à choisir le candidat, il surestime donc son niveau : un candidat chanceux peut le gagner sans être meilleur.

Deux points de départ sont possibles, depuis le bouton « Réinitialiser » :

- **Valeurs classiques** : une évaluation écrite à la main, que l'entraînement affine. Les progrès sont lents et beaucoup de candidats sont rejetés, c'est normal.
- **Zéro absolu** : le moteur ne connaît que les règles. Les premières générations progressent de plusieurs centaines d'ELO, et on voit apparaître la valeur des pièces et les bonnes cases.

## Le réseau de neurones (NNUE)

Dans « Réinitialiser », choisis **Passer au réseau de neurones**. Ton champion actuel et l'historique sont conservés : un réseau va apprendre à battre ce champion.

- **Architecture** : 768 entrées (2 couleurs × 6 pièces × 64 cases), puis 2 × 128 neurones cachés (la position vue par les Blancs et vue par les Noirs), puis une sortie qui donne l'évaluation. Environ 99 000 paramètres.
- **Rapide dans la recherche** : un coup ne change que 2 à 4 entrées, donc on met à jour les neurones cachés au lieu de tout recalculer. Les calculs se font en nombres entiers.
- **Données** : le champion joue contre lui-même. Chaque position calme est gardée avec le score de la recherche et le résultat de la partie. Le réseau apprend à prédire un mélange des deux (75 % score, 25 % résultat).
- **Premier entraînement** : il faut d'abord accumuler 400 000 positions, soit quelques minutes. Ensuite, chaque génération ajoute des parties et affine le réseau. Jusqu'à 2 millions de positions sont gardées en mémoire.
- **Validation** : une position sur 20 n'est jamais montrée au réseau pendant l'apprentissage. Son erreur (en pointillés sur le graphique) montre si le réseau généralise ou s'il apprend par cœur.
- **Adoption** : comme pour la formule classique, le réseau doit gagner le match de sélection puis la vérification.

Le réseau champion est enregistré dans `data/nnue.bin`.

## Crédits

Les pièces « Staunty » sont de sadsnake1, sous licence [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Elles proviennent de [lichess](https://github.com/lichess-org/lila). Cette licence interdit l'usage commercial.

Les tables de position de départ s'inspirent de la *Simplified Evaluation Function* de Tomasz Michniewski ([chessprogramming.org](https://www.chessprogramming.org/Simplified_Evaluation_Function)).
