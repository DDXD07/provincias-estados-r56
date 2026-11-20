# Provincias como Estados (Road to 56)

Mod para **Hearts of Iron IV 1.19** + **[The Road to 56](https://steamcommunity.com/sharedfiles/filedetails/?id=820260968)** en el que cada provincia terrestre del mapa es su propio estado (10 278 estados), con el nombre de su ciudad o municipio real.

## Qué hace
- Divide los 1 143 estados de Road to 56 en estados de una sola provincia. La provincia principal (la de mayor punto de victoria) conserva el ID original, así que capitales, focos y eventos siguen funcionando.
- Reparte manpower, recursos, fábricas y slots sin cambiar los totales de cada estado original.
- Expande `transfer_state`, `add_state_core/claim`, listas `states = { }` y bloques de núcleos/reclamaciones de Road to 56 y del juego base para que cubran todas las provincias del estado original.
- Nombra cada provincia con la ciudad más poblada que cae dentro de ella en el mundo real (provincias con punto de victoria conservan el nombre del juego). Reporte completo en `_generador/nombres_provincias.csv`.

## Instalación
1. Copia esta carpeta a `Documents/Paradox Interactive/Hearts of Iron IV/mod/provincias_estados_r56`.
2. Crea `mod/provincias_estados_r56.mod` con el contenido de `descriptor.mod` más la línea
   `path="C:/Users/<tu usuario>/Documents/Paradox Interactive/Hearts of Iron IV/mod/provincias_estados_r56"`.
3. En el launcher, ponlo **debajo de The Road to 56** en el playset y empieza una partida nueva.

## Regenerar (tras actualizaciones de Road to 56)
Requiere Python 3 con `numpy` y `pillow`. Ver `_generador/LEEME.txt`.

## Avisos
- Muy pesado para el motor: espera un juego más lento.
- Contiene archivos derivados de The Road to 56 y de Hearts of Iron IV (© sus autores / Paradox Interactive). Datos de ciudades: [GeoNames](https://www.geonames.org/), CC BY 4.0.
