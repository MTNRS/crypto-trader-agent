# Crypto Trader Agent

Repositorio independiente del agente de trading de OpenJarvis.

## Contenido

- `src/openjarvis/agents/crypto_trader.py`: implementación del agente.
- `src/openjarvis/agents/templates/crypto_trader.toml`: plantilla de configuración.
- `tests/agents/test_crypto_trader.py`: pruebas del agente.

## Integración

El agente usa interfaces de OpenJarvis (`BaseAgent`, `AgentRegistry` y las
herramientas de wallet). Para ejecutarlo, instala o añade al `PYTHONPATH` los
repositorios `jarvis-core`, `jarvis-agents` y `jarvis-integrations` compatibles.

No guardes claves, semillas o credenciales de exchanges en este repositorio.
Configúralas mediante variables de entorno o el almacén seguro de OpenJarvis.
