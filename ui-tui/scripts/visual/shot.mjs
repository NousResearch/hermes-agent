// Screenshot /tmp/tui-visual.html with the repo's Electron (offscreen).
import { app, BrowserWindow } from 'electron'
import { writeFileSync } from 'fs'

app.disableHardwareAcceleration()

app.whenReady().then(async () => {
  const win = new BrowserWindow({
    height: 2100,
    show: false,
    webPreferences: { offscreen: true },
    width: 1500
  })

  await win.loadFile('/tmp/tui-visual.html')
  await new Promise(r => setTimeout(r, 700))

  const image = await win.webContents.capturePage()

  writeFileSync('/tmp/tui-visual.png', image.toPNG())
  console.log('wrote /tmp/tui-visual.png')
  app.quit()
})
