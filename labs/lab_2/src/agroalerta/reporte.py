def contar_riesgos(
    sensores: list,
    lecturas: dict[str, list[float]],
) -> dict[str, int]:
    conteo: dict[str, int] = {}
    for sensor in sensores:
        valores = lecturas.get(sensor.nombre, [])
        conteo[sensor.nombre] = sum(
            1 for valor in valores if sensor.es_riesgo(valor)
        )
    return conteo
