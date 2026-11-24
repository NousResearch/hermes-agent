// Screenshot the render.tsx output with the workspace's Electron (offscreen).
import { app, BrowserWindow } from 'electron'
import { writeFileSync } from 'fs'
import { join } from 'path'

import { visualOutDir } from './paths.mjs'

app.disableHardwareAcceleration()

app.whenReady().then(async () => {
  const win = new BrowserWindow({
    height: 2100,
    show: false,
    webPreferences: { offscreen: true },
    width: 1500
  })

  const outDir = visualOutDir()

  await win.loadFile(join(outDir, 'tui-visual.html'))
  await new Promise(r => setTimeout(r, 700))

  const image = await win.webContents.capturePage()
  const outFile = join(outDir, 'tui-visual.png')

  writeFileSync(outFile, image.toPNG())
  console.log(`wrote ${outFile}`)
  app.quit()
})
