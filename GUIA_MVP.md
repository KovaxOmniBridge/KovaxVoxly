# Guia rápido: ativar e testar o MVP do Voxly

Este guia é focado em **subir o MVP localmente** e validar o fluxo completo:
**microfone -> limpeza de ruído -> VAD/diarização -> transcrição Whisper**.

## 1) Pré-requisitos

## Sistema operacional
- **Windows 10/11** (recomendado, pois o projeto usa FlexASIO e configurações WASAPI no fluxo atual).

## Ferramentas
- **Rust** (toolchain estável) e `cargo` no PATH.
- **Python 3.10+** no PATH.
- **Git** (opcional, para atualizar o projeto).

## Arquivos de modelo (na raiz do projeto)
Confirme que estes arquivos existem:
- `silero_vad.onnx`
- `segmentation-3.0.onnx`
- `enc.onnx`
- `erb_dec.onnx`
- `df_dec.onnx`
- `FlexASIO-1.10b.exe`

> Sem esses arquivos, o MVP não inicializa corretamente.

## 2) Preparar o serviço de transcrição (Python)

No diretório do projeto:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install faster-whisper websockets numpy
```

Inicie o servidor Whisper:

```bash
python bridge/transcriber.py
```

Ao subir, você deve ver logs como:
- carregamento do modelo (`large-v3-turbo`),
- servidor WebSocket em `127.0.0.1:8000`.

> O projeto está configurado para **CPU-only** (`device="cpu"`, `compute_type="int8"`).


### Importante sobre o `large-v3-turbo` (primeira execução)
Quando você roda `python bridge/transcriber.py` e ele fica em **"Carregando Whisper (large-v3-turbo)..."**, normalmente não é travamento: ele está **baixando e/ou convertendo o modelo na primeira vez**.

- O download pode levar vários minutos (dependendo da internet e disco).
- O modelo fica em cache local (normalmente em `C:\Users\<seu_usuario>\.cache\huggingface\hub`).
- Nas próximas execuções, inicializa muito mais rápido.

Se você quiser validar rapidamente que o servidor sobe, troque temporariamente no `bridge/transcriber.py`:

```python
MODEL_SIZE = "small"
```

Depois volte para `large-v3-turbo`.

### Preciso instalar manualmente o modelo?
**Não obrigatoriamente.** Instalando `faster-whisper`, o modelo é baixado automaticamente na primeira execução.

Se preferir controle total (ou pré-baixar antes), siga o passo a passo abaixo.

### Passo a passo (instalação/uso do Faster-Whisper no projeto)

1. Criar/ativar ambiente virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Atualizar `pip` e instalar dependências:

```bash
python -m pip install --upgrade pip
pip install faster-whisper websockets numpy
```

3. (Opcional) Fixar versões para estabilidade:

```bash
pip install "faster-whisper==1.0.3" "websockets==12.0" "numpy>=1.24,<3"
```

4. Pré-baixar o modelo **antes** de subir o servidor (opcional, mas recomendado):

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='int8'); print('modelo pronto em cache')"
```

5. Subir o servidor:

```bash
python bridge/transcriber.py
```

Se tudo estiver ok, você verá em sequência:
- mensagem de carregamento do modelo,
- mensagem de modelo carregado,
- mensagem de servidor WebSocket ativo na porta 8000.


## 3) Subir o app Rust (Voxly)

Em outro terminal, ainda na raiz do projeto:

```bash
cargo run
```

O aplicativo deve:
1. verificar/instalar FlexASIO (quando necessário),
2. configurar `FlexASIO.toml` no perfil do usuário,
3. conectar no WebSocket local do Python,
4. mostrar status do VAD no terminal.

## 3.1) Inicialização automática com interface (sem terminal manual)

Para iniciar tudo com um clique no Windows, use o arquivo:

```bash
Iniciar_Voxly.bat
```

Ele abre uma interface gráfica (`voxly_launcher_gui.py`) que:
- inicia automaticamente o `bridge/transcriber.py` no `.venv`,
- aguarda o Whisper subir na porta `8000`,
- executa `cargo run`,
- exibe logs separados de Python e Rust,
- mostra no painel central as transcrições ASR em tempo real.

> Se o `.venv` não existir, o launcher mostra erro e orienta os comandos de setup.

## 4) Roteiro de teste do MVP (passo a passo)

1. Com os dois processos rodando, fale frases curtas no microfone.
2. Observe no terminal Rust:
   - barra de atividade de voz (`[VAD]`),
   - identificação de locutor,
   - mensagem de frase finalizada após silêncio.
3. Após cada frase finalizada, confirme a transcrição retornada pelo serviço Python no terminal Rust.

Critérios mínimos de sucesso:
- captura de áudio ativa sem crash,
- VAD detectando fala e silêncio,
- pelo menos uma transcrição retornada para uma frase falada.

## 5) Problemas comuns (e como resolver rápido)

## Rust não compila em Linux com erro ALSA
- O projeto está orientado a Windows; em Linux, pode falhar por ausência de `alsa.pc`.
- Para validar o MVP com menor atrito, prefira Windows.

## Não conecta ao Whisper (`127.0.0.1:8000`)
- Confirme que `python bridge/transcriber.py` está rodando sem erro.
- Verifique se outra aplicação está usando a porta `8000`.

## Não aparece transcrição
- Fale frases de 2–5 segundos e faça pausa para o VAD fechar a frase.
- Verifique logs no Python para confirmar recebimento dos bytes de áudio.

## Qualidade ruim / latência alta
- Este projeto está em modo **CPU-only**; latência mais alta é esperada no `large-v3-turbo`.
- Para reduzir latência, troque temporariamente para `MODEL_SIZE = "small"` ou `"medium"` no `bridge/transcriber.py`.

## 6) Comandos úteis de validação rápida

```bash
# Validar sintaxe do serviço Python
python -m py_compile bridge/transcriber.py

# Verificar build Rust (em ambiente compatível)
cargo check
```

---
Se quiser, no próximo passo eu posso transformar este guia em um **checklist automatizado** (script) para validar pré-requisitos antes de executar o MVP.
