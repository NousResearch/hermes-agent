import AppKit
import CoreGraphics
import Foundation

func cgWindowInt(_ value: Any?) -> Int? {
    if let num = value as? NSNumber { return num.intValue }
    if let intVal = value as? Int { return intVal }
    return nil
}

func cgWindowDouble(_ value: Any?) -> Double? {
    if let num = value as? NSNumber { return num.doubleValue }
    if let doubleVal = value as? Double { return doubleVal }
    if let intVal = value as? Int { return Double(intVal) }
    return nil
}

let workspace = NSWorkspace.shared
let app = workspace.frontmostApplication
let pid = app?.processIdentifier ?? 0
let appName = app?.localizedName ?? ""
let bundleID = app?.bundleIdentifier ?? ""
let bundleName = app?.bundleURL?.deletingPathExtension().lastPathComponent ?? ""

let rawInfo = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] ?? []

let windows: [[String: Any]] = rawInfo.compactMap { item in
    guard cgWindowInt(item[kCGWindowOwnerPID as String]) == Int(pid) else { return nil }
    let layer = cgWindowInt(item[kCGWindowLayer as String]) ?? 0
    guard layer == 0 else { return nil }
    let alpha = cgWindowDouble(item[kCGWindowAlpha as String]) ?? 1.0
    guard alpha > 0 else { return nil }
    guard let bounds = item[kCGWindowBounds as String] as? [String: Any] else { return nil }
    let width = cgWindowDouble(bounds["Width"]) ?? 0
    let height = cgWindowDouble(bounds["Height"]) ?? 0
    guard width > 1 && height > 1 else { return nil }
    return [
        "window_title": (item[kCGWindowName as String] as? String) ?? "",
        "window_id": cgWindowInt(item[kCGWindowNumber as String]) ?? NSNull(),
        "window_bounds": [
            "x": Int(cgWindowDouble(bounds["X"]) ?? 0),
            "y": Int(cgWindowDouble(bounds["Y"]) ?? 0),
            "width": Int(width),
            "height": Int(height),
        ],
    ]
}

var payload: [String: Any] = [
    "app_name": appName,
    "bundle_id": bundleID,
    "bundle_name": bundleName,
    "process_id": Int(pid),
    "window_title": "",
    "window_id": NSNull(),
    "window_bounds": NSNull(),
    "windows": windows,
]

if let first = windows.first {
    payload["window_title"] = first["window_title"] ?? ""
    payload["window_id"] = first["window_id"] ?? NSNull()
    payload["window_bounds"] = first["window_bounds"] ?? NSNull()
}

let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
FileHandle.standardOutput.write(data)
