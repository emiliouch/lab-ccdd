import argparse
from pathlib import Path

from src.agroalerta.datos import cargar_lecturas
from src.agroalerta.reporte import contar_riesgos
from src.agroalerta.sensores import (
    SensorHumedad,
    SensorTemperatura,
    SensorViento,
)


def main():
    parser = argparse.ArgumentParser(description="AgroAlerta")
    parser.add_argument("--fecha", default="2026-06-15")
    args = parser.parse_args()

    sensores = [
        SensorTemperatura(0, 40),
        SensorViento(25),
        SensorHumedad(85),
    ]

    ruta = Path("data/lecturas.csv")
    lecturas = cargar_lecturas(ruta, args.fecha)
    conteo = contar_riesgos(sensores, lecturas)

    total = sum(conteo.values())
    print(f"Estación Parcela Norte — {args.fecha}")
    for sensor in sensores:
        cantidad = conteo.get(sensor.nombre, 0)
        print(f"{sensor.nombre.capitalize():<14} {cantidad} lecturas en riesgo")
    print(f"\nTotal: {total} situaciones de riesgo")


if __name__ == "__main__":
    main()
