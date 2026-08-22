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

  // The scene grid can grow when new skins are added. Size the offscreen
  // surface from the rendered document instead of silently clipping rows at
  // the old fixed 2100px viewport (which hid the fifth mono scene).
  const pageSize = await win.webContents.executeJavaScript(`({
    height: Math.ceil(document.documentElement.scrollHeight),
    width: Math.ceil(document.documentElement.scrollWidth)
  })`)

  win.setContentSize(pageSize.width, pageSize.height)
  await new Promise(r => setTimeout(r, 100))

  const image = await win.webContents.capturePage({ height: pageSize.height, width: pageSize.width, x: 0, y: 0 })
  const outFile = join(outDir, 'tui-visual.png')

  writeFileSync(outFile, image.toPNG())
  console.log(`wrote ${outFile}`)
  app.quit()
})
