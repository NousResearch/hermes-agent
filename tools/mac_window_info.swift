import AppKit
import ApplicationServices
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

func axAttribute(_ element: AXUIElement, _ attribute: CFString) -> CFTypeRef? {
    var value: CFTypeRef?
    let error = AXUIElementCopyAttributeValue(element, attribute, &value)
    guard error == .success else { return nil }
    return value
}

func axString(_ element: AXUIElement, _ attribute: CFString) -> String? {
    guard let value = axAttribute(element, attribute) else { return nil }
    if let string = value as? String {
        return string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : string
    }
    if let number = value as? NSNumber {
        return number.stringValue
    }
    return nil
}

func axBool(_ element: AXUIElement, _ attribute: CFString) -> Bool? {
    guard let value = axAttribute(element, attribute) else { return nil }
    if let boolValue = value as? Bool { return boolValue }
    if let number = value as? NSNumber {
        if CFGetTypeID(number) == CFBooleanGetTypeID() {
            return number.boolValue
        }
    }
    return nil
}

func axElement(_ element: AXUIElement, _ attribute: CFString) -> AXUIElement? {
    guard let value = axAttribute(element, attribute) else { return nil }
    guard CFGetTypeID(value) == AXUIElementGetTypeID() else { return nil }
    return unsafeBitCast(value, to: AXUIElement.self)
}

func axElements(_ element: AXUIElement, _ attribute: CFString) -> [AXUIElement] {
    guard let value = axAttribute(element, attribute) else { return [] }
    if let elements = value as? [AXUIElement] {
        return elements
    }
    guard let raw = value as? [Any] else { return [] }
    return raw.compactMap { item in
        guard CFGetTypeID(item as CFTypeRef) == AXUIElementGetTypeID() else { return nil }
        return unsafeBitCast(item as CFTypeRef, to: AXUIElement.self)
    }
}

func axPoint(_ element: AXUIElement, _ attribute: CFString) -> CGPoint? {
    guard let value = axAttribute(element, attribute) else { return nil }
    guard CFGetTypeID(value) == AXValueGetTypeID() else { return nil }
    let axValue = unsafeBitCast(value, to: AXValue.self)
    guard AXValueGetType(axValue) == .cgPoint else { return nil }
    var point = CGPoint.zero
    return AXValueGetValue(axValue, .cgPoint, &point) ? point : nil
}

func axSize(_ element: AXUIElement, _ attribute: CFString) -> CGSize? {
    guard let value = axAttribute(element, attribute) else { return nil }
    guard CFGetTypeID(value) == AXValueGetTypeID() else { return nil }
    let axValue = unsafeBitCast(value, to: AXValue.self)
    guard AXValueGetType(axValue) == .cgSize else { return nil }
    var size = CGSize.zero
    return AXValueGetValue(axValue, .cgSize, &size) ? size : nil
}

func axFrame(_ element: AXUIElement) -> [String: Int]? {
    guard let position = axPoint(element, kAXPositionAttribute as CFString),
          let size = axSize(element, kAXSizeAttribute as CFString) else { return nil }
    return [
        "x": Int(position.x.rounded()),
        "y": Int(position.y.rounded()),
        "width": Int(size.width.rounded()),
        "height": Int(size.height.rounded()),
    ]
}

func axValueScalar(_ element: AXUIElement, _ attribute: CFString) -> Any? {
    guard let value = axAttribute(element, attribute) else { return nil }
    if let string = value as? String {
        return string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : string
    }
    if let number = value as? NSNumber {
        if CFGetTypeID(number) == CFBooleanGetTypeID() {
            return number.boolValue
        }
        let doubleValue = number.doubleValue
        let intValue = number.intValue
        return Double(intValue) == doubleValue ? intValue : doubleValue
    }
    return nil
}

func axActionNames(_ element: AXUIElement) -> [String] {
    var names: CFArray?
    let error = AXUIElementCopyActionNames(element, &names)
    guard error == .success, let names else { return [] }
    return (names as? [String] ?? []).filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
}

let maxAXDepth = 5
let maxAXChildrenPerNode = 60
let maxAXNodes = 300

func accessibilityNode(_ element: AXUIElement, depth: Int, remaining: inout Int) -> [String: Any]? {
    guard remaining > 0 else { return nil }
    remaining -= 1

    var node: [String: Any] = [:]
    if let role = axString(element, kAXRoleAttribute as CFString) { node["role"] = role }
    if let subrole = axString(element, kAXSubroleAttribute as CFString) { node["subrole"] = subrole }
    if let title = axString(element, kAXTitleAttribute as CFString) { node["title"] = title }
    if let description = axString(element, kAXDescriptionAttribute as CFString) { node["description"] = description }
    if let value = axValueScalar(element, kAXValueAttribute as CFString) { node["value"] = value }
    if let enabled = axBool(element, kAXEnabledAttribute as CFString) { node["enabled"] = enabled }
    if let focused = axBool(element, kAXFocusedAttribute as CFString) { node["focused"] = focused }
    if let frame = axFrame(element) { node["frame"] = frame }

    let actions = axActionNames(element)
    if !actions.isEmpty { node["actions"] = actions }

    var children: [[String: Any]] = []
    if depth < maxAXDepth && remaining > 0 {
        for child in axElements(element, kAXChildrenAttribute as CFString).prefix(maxAXChildrenPerNode) {
            if let childNode = accessibilityNode(child, depth: depth + 1, remaining: &remaining) {
                children.append(childNode)
            }
            if remaining <= 0 { break }
        }
    }
    node["children"] = children

    if node.isEmpty || (node.keys.count == 1 && node["children"] != nil && children.isEmpty) {
        return nil
    }
    return node
}

func accessibilityTreeForFrontmostApp(pid: pid_t) -> [[String: Any]] {
    guard pid > 0 else { return [] }
    guard AXIsProcessTrusted() else { return [] }
    let appElement = AXUIElementCreateApplication(pid)
    let windowElement = axElement(appElement, kAXFocusedWindowAttribute as CFString)
        ?? axElement(appElement, kAXMainWindowAttribute as CFString)
    guard let windowElement else { return [] }
    var remaining = maxAXNodes
    guard let root = accessibilityNode(windowElement, depth: 0, remaining: &remaining) else { return [] }
    return [root]
}

let workspace = NSWorkspace.shared
let app = workspace.frontmostApplication
let pid = app?.processIdentifier ?? 0
let appName = app?.localizedName ?? ""
let bundleID = app?.bundleIdentifier ?? ""
let bundleURL = app?.bundleURL
let bundleName = bundleURL?.deletingPathExtension().lastPathComponent ?? ""
let bundlePath = bundleURL?.path ?? ""

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
    "bundle_path": bundlePath,
    "process_id": Int(pid),
    "window_title": "",
    "window_id": NSNull(),
    "window_bounds": NSNull(),
    "windows": windows,
    "accessibility_tree": accessibilityTreeForFrontmostApp(pid: pid),
]

if let first = windows.first {
    payload["window_title"] = first["window_title"] ?? ""
    payload["window_id"] = first["window_id"] ?? NSNull()
    payload["window_bounds"] = first["window_bounds"] ?? NSNull()
}

let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
FileHandle.standardOutput.write(data)
