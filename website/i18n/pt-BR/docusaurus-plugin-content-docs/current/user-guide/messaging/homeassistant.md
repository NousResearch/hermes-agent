---
title: Home Assistant
description: Controle sua casa inteligente com o Hermes Agent via integração Home Assistant.
sidebar_label: Home Assistant
sidebar_position: 5
---

# Home Assistant Integration

O Hermes Agent integra com [Home Assistant](https://www.home-assistant.io/) de duas formas:

1. **Plataforma gateway** — assina mudanças de estado em tempo real via WebSocket e responde a eventos
2. **Ferramentas de casa inteligente** — quatro ferramentas invocáveis pelo LLM para consultar e controlar dispositivos via REST API

## Setup {#setup}

### 1. Create a Long-Lived Access Token

1. Abra sua instância Home Assistant
2. Vá ao seu **Profile** (clique no seu nome na barra lateral)
3. Role até **Long-Lived Access Tokens**
4. Clique em **Create Token**, dê um nome como "Hermes Agent"
5. Copie o token

### 2. Configure Environment Variables

```bash
# Adicione em ~/.hermes/.env

# Obrigatório: seu Long-Lived Access Token
HASS_TOKEN=your-long-lived-access-token

# Opcional: URL do HA (padrão: http://homeassistant.local:8123)
HASS_URL=http://192.168.1.100:8123
```

:::info
O toolset `homeassistant` é habilitado automaticamente quando `HASS_TOKEN` está definido. Tanto a plataforma gateway quanto as ferramentas de controle de dispositivos ativam a partir deste único token.
:::

### 3. Start the Gateway

```bash
hermes gateway
```

O Home Assistant aparecerá como plataforma conectada junto com quaisquer outras plataformas de mensagens (Telegram, Discord, etc.).

## Available Tools {#available-tools}

O Hermes Agent registra quatro ferramentas para controle de casa inteligente:

### `ha_list_entities`

Lista entidades Home Assistant, opcionalmente filtradas por domínio ou área.

**Parameters:**
- `domain` *(opcional)* — Filtrar por domínio de entidade: `light`, `switch`, `climate`, `sensor`, `binary_sensor`, `cover`, `fan`, `media_player`, etc.
- `area` *(opcional)* — Filtrar por nome de área/cômodo (corresponde a friendly names): `living room`, `kitchen`, `bedroom`, etc.

**Example:**
```
List all lights in the living room
```

Retorna IDs de entidade, estados e friendly names.

### `ha_get_state`

Obtém estado detalhado de uma única entidade, incluindo todos os atributos (brilho, cor, setpoint de temperatura, leituras de sensor, etc.).

**Parameters:**
- `entity_id` *(obrigatório)* — A entidade a consultar, ex.: `light.living_room`, `climate.thermostat`, `sensor.temperature`

**Example:**
```
What's the current state of climate.thermostat?
```

Retorna: estado, todos os atributos, timestamps de last changed/updated.

### `ha_list_services`

Lista serviços disponíveis (ações) para controle de dispositivos. Mostra quais ações podem ser executadas em cada tipo de dispositivo e quais parâmetros aceitam.

**Parameters:**
- `domain` *(opcional)* — Filtrar por domínio, ex.: `light`, `climate`, `switch`

**Example:**
```
What services are available for climate devices?
```

### `ha_call_service`

Chama um serviço Home Assistant para controlar um dispositivo.

**Parameters:**
- `domain` *(obrigatório)* — Domínio do serviço: `light`, `switch`, `climate`, `cover`, `media_player`, `fan`, `scene`, `script`
- `service` *(obrigatório)* — Nome do serviço: `turn_on`, `turn_off`, `toggle`, `set_temperature`, `set_hvac_mode`, `open_cover`, `close_cover`, `set_volume_level`
- `entity_id` *(opcional)* — Entidade alvo, ex.: `light.living_room`
- `data` *(opcional)* — Parâmetros adicionais como objeto JSON

**Examples:**

```
Turn on the living room lights
→ ha_call_service(domain="light", service="turn_on", entity_id="light.living_room")
```

```
Set the thermostat to 22 degrees in heat mode
→ ha_call_service(domain="climate", service="set_temperature",
    entity_id="climate.thermostat", data={"temperature": 22, "hvac_mode": "heat"})
```

```
Set living room lights to blue at 50% brightness
→ ha_call_service(domain="light", service="turn_on",
    entity_id="light.living_room", data={"brightness": 128, "color_name": "blue"})
```

## Gateway Platform: Real-Time Events {#gateway-platform-real-time-events}

O adapter gateway Home Assistant conecta via WebSocket e assina eventos `state_changed`. Quando o estado de um dispositivo muda e corresponde aos seus filtros, é encaminhado ao agente como mensagem.

### Event Filtering

:::warning Required Configuration
Por padrão, **nenhum evento é encaminhado**. Você deve configurar pelo menos um de `watch_domains`, `watch_entities` ou `watch_all` para receber eventos. Sem filtros, um aviso é registrado na inicialização e todas as mudanças de estado são descartadas silenciosamente.
:::

Configure quais eventos o agente vê em `~/.hermes/config.yaml` na seção `extra` da plataforma Home Assistant:

```yaml
platforms:
  homeassistant:
    enabled: true
    extra:
      watch_domains:
        - climate
        - binary_sensor
        - alarm_control_panel
        - light
      watch_entities:
        - sensor.front_door_battery
      ignore_entities:
        - sensor.uptime
        - sensor.cpu_usage
        - sensor.memory_usage
      cooldown_seconds: 30
```

| Setting | Default | Description |
|---------|---------|-------------|
| `watch_domains` | *(nenhum)* | Observar apenas estes domínios de entidade (ex.: `climate`, `light`, `binary_sensor`) |
| `watch_entities` | *(nenhum)* | Observar apenas estes IDs de entidade específicos |
| `watch_all` | `false` | Defina `true` para receber **todas** as mudanças de estado (não recomendado na maioria dos setups) |
| `ignore_entities` | *(nenhum)* | Sempre ignorar estas entidades (aplicado antes dos filtros de domínio/entidade) |
| `cooldown_seconds` | `30` | Segundos mínimos entre eventos para a mesma entidade |

:::tip
Comece com um conjunto focado de domínios — `climate`, `binary_sensor` e `alarm_control_panel` cobrem as automações mais úteis. Adicione mais conforme necessário. Use `ignore_entities` para suprimir sensores ruidosos como temperatura de CPU ou contadores de uptime.
:::

### Event Formatting

Mudanças de estado são formatadas como mensagens legíveis conforme o domínio:

| Domain | Format |
|--------|--------|
| `climate` | "HVAC mode changed from 'off' to 'heat' (current: 21, target: 23)" |
| `sensor` | "changed from 21°C to 22°C" |
| `binary_sensor` | "triggered" / "cleared" |
| `light`, `switch`, `fan` | "turned on" / "turned off" |
| `alarm_control_panel` | "alarm state changed from 'armed_away' to 'triggered'" |
| *(other)* | "changed from 'old' to 'new'" |

### Agent Responses

Mensagens outbound do agente são entregues como **notificações persistentes Home Assistant** (via `persistent_notification.create`). Elas aparecem no painel de notificações do HA com o título "Hermes Agent".

### Connection Management

- **WebSocket** com heartbeat de 30 segundos para eventos em tempo real
- **Reconexão automática** com backoff: 5s → 10s → 30s → 60s
- **REST API** para notificações outbound (sessão separada para evitar conflitos WebSocket)
- **Autorização** — eventos HA são sempre autorizados (sem allowlist de usuário necessária, já que `HASS_TOKEN` autentica a conexão)

## Security {#security}

As ferramentas Home Assistant impõem restrições de segurança:

:::warning Blocked Domains
Os seguintes domínios de serviço estão **bloqueados** para evitar execução arbitrária de código no host HA:

- `shell_command` — comandos shell arbitrários
- `command_line` — sensores/switches que executam comandos
- `python_script` — execução Python scriptada
- `pyscript` — integração de scripting mais ampla
- `hassio` — controle de addon, shutdown/reboot do host
- `rest_command` — requisições HTTP do servidor HA (vetor SSRF)

Tentar chamar serviços nesses domínios retorna erro.
:::

IDs de entidade são validados contra o padrão `^[a-z_][a-z0-9_]*\.[a-z0-9_]+$` para evitar ataques de injeção.

## Example Automations {#example-automations}

### Morning Routine

```
User: Start my morning routine

Agent:
1. ha_call_service(domain="light", service="turn_on",
     entity_id="light.bedroom", data={"brightness": 128})
2. ha_call_service(domain="climate", service="set_temperature",
     entity_id="climate.thermostat", data={"temperature": 22})
3. ha_call_service(domain="media_player", service="turn_on",
     entity_id="media_player.kitchen_speaker")
```

### Security Check

```
User: Is the house secure?

Agent:
1. ha_list_entities(domain="binary_sensor")
     → verifica sensores de porta/janela
2. ha_get_state(entity_id="alarm_control_panel.home")
     → verifica status do alarme
3. ha_list_entities(domain="lock")
     → verifica estados das fechaduras
4. Reports: "All doors closed, alarm is armed_away, all locks engaged."
```

### Reactive Automation (via Gateway Events)

Quando conectado como plataforma gateway, o agente pode reagir a eventos:

```
[Home Assistant] Front Door: triggered (was cleared)

Agent automatically:
1. ha_get_state(entity_id="binary_sensor.front_door")
2. ha_call_service(domain="light", service="turn_on",
     entity_id="light.hallway")
3. Sends notification: "Front door opened. Hallway lights turned on."
```

## Troubleshooting {#troubleshooting}

**Variáveis de ambiente não aplicadas.**
O adapter lê credenciais de `~/.hermes/.env` (mesclado automaticamente na inicialização) ou de `config.yaml`. Confira se o arquivo está no home do perfil Hermes ativo e se não há aspas extras na URL/token. Reinicie o gateway após editar — mudanças de env só aplicam na inicialização do processo.


**Falha de auth REST (`401 Unauthorized`).**
O token deve ser um *Long-Lived Access Token* criado na página de perfil do usuário HA (**Profile → Security → Long-lived access tokens**). Tokens de sessão de curta duração da UI não funcionam. Verifique também se a URL base inclui esquema e porta (ex. `http://homeassistant.local:8123`) e é acessível do host que roda o Hermes — `curl -H "Authorization: Bearer <token>" <url>/api/` deve retornar `{"message": "API running."}`.
