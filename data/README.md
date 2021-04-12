## Datos 

Tenemos dos archivos con los datos etiquetados

- `articles.json`: los artículos
- `comments.json`: los comentarios

## articles.json

Es una lista de artículos con los siguientes campso

- `tweet_id`: Id del Tweet del medio periodístico
- `title`: Título de la nota periodística

Ejemplo: "Coronavirus en Argentina: Cristina Kirchner vuelve de Cuba junto a su hija Florencia y permanecerá aislada por 14 días"
- `tweet_text`: Texto del tweet original

"Coronavirus en Argentina: Cristina Kirchner vuelve de Cuba junto a su hija Florencia y permanecerá aislada por 14 días https://t.co/ltgNhDGTlZ"
- `body`: Cuerpo entero de la noticia

Ejemplo: «En plena cuarentena, la vicepresidenta Cristina Fernández de Kirchner llegará este domingo de Cuba junto a su hija Florencia, quien vuelve a la Argentina tras un año de tratamientos médicos en ese país. Tal como anunció en sus redes sociales, la ex mandataria se mantendrá durante dos semanas en aislamiento. "Si bien Cuba no es un país de riesgo, al llegar cumpliré con los 14 días de aislamiento", escribió el viernes en cuenta de Twitter. En su entorno confirmaron que, aunque por su carácter de vicepresidenta de la ... (sigue)»

- `news`: Nombre de la cuenta de Twitter del medio periodístico
Ej: "clarincom"
- `date`: Fecha en formato ISO
Ejemplo: `2020-03-22T01:08:09.000000Z`

## comments.json 

Puede llamarse `comments_not_anon.json` si tiene el `tweet_id`. 

- `id`: Numero de identificación interno
- `tweet_id`:1241513939499876400
- `text`: Texto del comentario
Ejemplo: "@usuario Momento oportuno para hacer esa movida!!!!..ahora que la JUSTICIA está en veda sanitaria..!!!!.."
- `article_id`: Tweet id del artículo en el que se hizo el comentario
- `HATE`: Identificadores de etiquetadores que marcaron el comentario como odioso
- `CALLS`: Identificadores de etiquetadores que marcaron el comentario como odioso
- `WOMEN`: Identificadores de etiquetadores que marcaron el comentario como MUJER
- `LGBTI`: Identificadores de etiquetadores que marcaron el comentario como LGBTI
- `RACISM`: Identificadores de etiquetadores que marcaron el comentario como RACISMO
- `CLASS`: Identificadores de etiquetadores que marcaron el comentario como POBREZA
- `POLITICS`: Identificadores de etiquetadores que marcaron el comentario como POLITICA
- `DISABLED`: Identificadores de etiquetadores que marcaron el comentario como DISCAPACIDAD
- `APPEARANCE`: Identificadores de etiquetadores que marcaron el comentario como APARIENCIA
- `CRIMINAL`: Identificadores de etiquetadores que marcaron el comentario como CRIMINAL

Todos los identificadores de los anotadores están anonimizados

## train.json

Agrego una separación de estos datos con una asignación binaria a cada etiqueta correspondiente.