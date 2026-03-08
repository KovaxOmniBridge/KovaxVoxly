# Relatório Geral do Projeto Voxly

## 1) Visão geral
O **Voxly** é um pipeline de áudio em tempo real, escrito em Rust, com foco em:
- captura de microfone,
- limpeza de ruído (DeepFilter em ONNX),
- detecção de fala (VAD com Silero),
- diarização de locutor (modelo de segmentação),
- transcrição via microserviço Python (faster-whisper) por WebSocket.

Na prática, o aplicativo funciona como uma ponte de áudio com processamento de IA e saída local, além de imprimir transcrições por locutor no terminal.

## 2) Arquitetura atual

### Camada Rust (núcleo em tempo real)
- **`src/main.rs`**: inicializa ambiente e inicia o pipeline principal.
- **`src/driver.rs`**: prepara ambiente de driver (FlexASIO), gera `FlexASIO.toml` com dispositivos padrão e latência baixa.
- **`src/audio.rs`**: orquestra captura/reprodução com `cpal`, aplica denoising, resampling simplificado para 16 kHz, coordena VAD+diarização em thread dedicada e envia frases finalizadas ao serviço Whisper via WebSocket.
- **`src/engine/*`**:
  - `denoiser.rs`: inferência ONNX em 3 estágios (`enc.onnx`, `erb_dec.onnx`, `df_dec.onnx`);
  - `vad.rs`: inferência Silero VAD e manutenção de estado recorrente;
  - `diarizer.rs`: inferência de segmentação e identificação dominante de locutor;
  - `buffers.rs`: buffer de pré-roll para não cortar início de frase.

### Camada Python (ASR)
- **`bridge/transcriber.py`**:
  - sobe servidor WebSocket local (`127.0.0.1:8000`),
  - recebe áudio PCM float32 (16 kHz),
  - transcreve com `faster-whisper` (`large-v3-turbo`) em modo CPU-only (`int8`),
  - retorna resposta em JSON com texto e latência.

## 3) Pontos fortes
- **Pipeline completo ponta a ponta** (captura → limpeza → VAD/diarização → ASR).
- **Separação de responsabilidades razoável** entre módulos de engine e bridge Python.
- **Uso de pré-roll** para melhorar UX e preservar o início das frases.
- **ASR Python configurado para CPU-only** (`int8`), alinhado ao cenário operacional atual.
- **Feedback visual em terminal** (barra de probabilidade de VAD e locutor atual).

## 4) Riscos e limitações identificadas

### 4.1 Portabilidade e acoplamento a Windows
- O fluxo de driver está claramente orientado a **Windows + FlexASIO**, com caminhos fixos de instalação e escrita em `USERPROFILE`.
- O projeto depende de binários/modelos colocados na raiz e paths relativos, o que reduz previsibilidade de deploy.

### 4.2 Robustez de runtime
- Há diversos `unwrap`/`expect` em pontos críticos de inferência e device setup; em produção isso pode encerrar o processo abruptamente.
- A task WebSocket do Rust tenta conectar em loop, mas ao cair durante envio/resposta faz `break` e não reconecta automaticamente.

### 4.3 Contrato de dados Rust ↔ Python
- O Python retorna JSON (`{"status":"success","text":"..."}`), mas o Rust imprime diretamente `Message::Text(txt)` sem parse.
- Isso gera saída “bruta” em vez de mensagem já tratada por campo.

### 4.4 Qualidade de áudio/resampling
- O resampling para 16 kHz é simplificado por “sample skipping” (`sample_step` e reset de posição), o que pode introduzir artefatos.
- Não há normalização/controle explícito de nível antes do ASR.

### 4.5 Observabilidade e operação
- Falta camada de logs estruturados, métricas e níveis configuráveis.
- Não há testes automatizados no repositório para garantir regressão mínima.
- Não há documentação de setup (README, dependências Python, versão de CUDA, como baixar modelos).

## 5) Estado de build/validação no ambiente atual
- **`cargo check` falhou** neste ambiente Linux por ausência de biblioteca de sistema `alsa` (`alsa.pc` não encontrado), bloqueando validação completa do binário Rust.
- **`python -m py_compile bridge/transcriber.py` passou**, indicando sintaxe Python válida.

## 6) Recomendações priorizadas

### Curto prazo (alto impacto / baixo esforço)
1. **Criar README operacional** com:
   - pré-requisitos Rust/Python,
   - dependências de sistema,
   - instruções de modelos ONNX,
   - passo a passo de execução (ordem Rust/Python).
2. **Padronizar contrato WS**:
   - parsear JSON no Rust,
   - imprimir apenas `text`, `elapsed_s`, `status` de forma limpa.
3. **Substituir `unwrap/expect` críticos** por tratamento de erro com contexto.
4. **Adicionar reconexão WS resiliente** após falha de envio/recebimento.

### Médio prazo
5. **Refatorar paths/configuração** para arquivo de config (`.toml/.yaml`) e variáveis de ambiente.
6. **Introduzir logs estruturados** (`tracing`/`env_logger`) e níveis por ambiente.
7. **Melhorar resampling** com biblioteca DSP dedicada para manter qualidade de transcrição.
8. **Adicionar smoke tests** (ex.: testes de contrato WS e testes unitários de buffers).

### Longo prazo
9. **Estratégia multiplataforma** (Windows/Linux/macOS) para captura/driver e packaging.
10. **Pipeline de CI** com checks de lint/build/test em matriz de SO quando possível.

## 7) Conclusão executiva
O projeto já demonstra um **MVP funcional tecnicamente ambicioso** para transcrição em tempo real com diarização e redução de ruído. O principal próximo passo é sair de “protótipo funcional” para “produto operável”: documentação, robustez de erros, contrato WS consistente, reconexão e testes mínimos. Com essas melhorias, a base atual tem bom potencial para evoluir para uma ferramenta estável de uso contínuo.
