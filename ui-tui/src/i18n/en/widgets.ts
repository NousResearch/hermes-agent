// 内置终端小组件的展示文案。
export const widgetsEn = {
  widget: {
    grid: {
      tokenStream:
        'Hermes streams tokens into the promoted cell while the grid reshapes around it. Cells are keyed by id, so promotion never resets a panel — history, cursors and tickers all survive the relayout. Row and column tracks re-solve to integer terminal cells on every change, spans bridge the gaps they cross, and dense auto-placement backfills the holes the promoted panel leaves behind.',
      uptime: (minutes: unknown, seconds: unknown) => `up ${minutes}m ${seconds}s`,
      ticks: (count: unknown) => `${count} ticks`,
      stream: {
        tokens: 'token stream',
        throughput: 'throughput',
        memory: 'memory',
        latency: 'latency',
        tools: 'tool feed'
      },
      missionControl: 'hermes mission control',
      main: (name: unknown) => `main: ${name}`,
      streams: 'streams',
      dimensions: (cols: unknown, rows: unknown) => `${cols}x${rows} grid`,
      zoomHint: 'arrows/hjkl switch cell · Esc/q back · Ctrl+C close',
      streamHint: 'arrows/hjkl focus · Enter promote · d dialog · Esc/q back · Ctrl+C close',
      controlsHint:
        'arrows/hjkl move · Enter zoom · d dialog · a areas · s streams · +/- cols · [] rows · g gap · p pad · n nest · q close',
      status: (gap: unknown, pad: unknown, nested: unknown, areas: unknown) =>
        `gap ${gap} · pad ${pad} · nested ${nested} · areas ${areas}`,
      auto: 'auto',
      areasOn: 'on (2fr first col · c1 spans rows · c2 spans cols)',
      parent: (cell: unknown) => `parent ${cell}`,
      nestedChild: 'nested child grid',
      help: 'open an interactive widget-grid demo overlay',
      usage: 'usage: /grid-test [cols]x[rows]  ·  /grid-test [cols] [rows]  ·  /grid-test streams',
      dialogBody: 'Dialog overlaid on top of /grid-test.\n\nBackdrop dims the grid behind.',
      dialogHint: 'Esc/q/Enter close',
      dialogTitle: 'Overlay primitive'
    },
    dialog: {
      usage:
        'usage: /dialog-test [zone]   zones: bottom, bottom-left, bottom-right, center, left, right, top, top-left, top-right',
      body: (zone: unknown) => `This is a viewport-level overlay with a backdrop.

Zone: ${zone}
Try: /dialog-test top-right · bottom · left · ...`,
      hint: 'Esc/q/Enter close · Ctrl+C close',
      title: 'Dialog primitive'
    },
    weather: {
      help: 'current conditions with themed ASCII art (Open-Meteo)',
      usage: 'usage: /weather [location]   (blank = geolocate by IP)',
      here: 'Here',
      title: 'Weather',
      error: (message: unknown) => `Could not load weather: ${message}`,
      unknown: 'Unknown conditions',
      feels: (temperature: unknown) => `(feels ${temperature})`,
      wind: (speed: unknown) => `wind ${speed}`,
      humidity: (percent: unknown) => `humidity ${percent}%`
    },
    unknownApp: (id: unknown) => `unknown widget app: ${id}`,
    usage: (id: unknown) => `usage: /${id}`,
    ticker: { help: 'fake 1-pip chart with a live sparkline', usage: 'usage: /ticker [symbol]' }
  }
}
