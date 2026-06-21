import Foundation
import ApplicationServices

struct Config {
    var pid: pid_t = -1
    var windowTitle: String?
    var windowIndex: Int = 0
    var windowBounds: [Double]? = nil
    var maxDepth: Int = 5
    var maxNodes: Int = 300
}

let coreAttributes = [
    "AXRole", "AXSubrole", "AXTitle", "AXDescription", "AXValue",
    "AXPlaceholderValue", "AXHelp", "AXIdentifier", "AXRoleDescription",
    "AXEnabled", "AXFocused", "AXPosition", "AXSize",
]

let childAttributes = [
    "AXChildren", "AXVisibleChildren", "AXRows", "AXColumns", "AXTabs",
    "AXContents", "AXSplitters", "AXToolbarButton",
]

func emitJSON(_ object: Any, exitCode: Int32 = 0) -> Never {
    do {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write("\n".data(using: .utf8)!)
    } catch {
        fputs("{\"ok\":false,\"error\":\"json_serialization_failed\"}\n", stderr)
        exit(1)
    }
    exit(exitCode)
}

func usage() -> Never {
    emitJSON([
        "ok": false,
        "error": "usage",
        "usage": "ax_geometry_helper --pid PID [--window-title TEXT] [--window-index N] [--window-bounds x,y,w,h] [--max-depth N] [--max-nodes N]",
    ], exitCode: 2)
}

func parseArgs() -> Config {
    var cfg = Config()
    var i = 1
    let args = CommandLine.arguments
    while i < args.count {
        let arg = args[i]
        func value() -> String {
            if i + 1 >= args.count { usage() }
            i += 1
            return args[i]
        }
        switch arg {
        case "--pid":
            cfg.pid = pid_t(Int32(value()) ?? -1)
        case "--window-title":
            cfg.windowTitle = value()
        case "--window-index":
            cfg.windowIndex = Int(value()) ?? 0
        case "--window-bounds":
            let parts = value().split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            if parts.count == 4 { cfg.windowBounds = parts }
        case "--max-depth":
            cfg.maxDepth = max(0, Int(value()) ?? cfg.maxDepth)
        case "--max-nodes":
            cfg.maxNodes = max(1, Int(value()) ?? cfg.maxNodes)
        default:
            usage()
        }
        i += 1
    }
    if cfg.pid <= 0 { usage() }
    return cfg
}

func copyAttribute(_ element: AXUIElement, _ name: String) -> (Bool, Any?, AXError) {
    var value: CFTypeRef?
    let err = AXUIElementCopyAttributeValue(element, name as CFString, &value)
    return (err == .success, value, err)
}

func actionNames(_ element: AXUIElement) -> [String] {
    var actions: CFArray?
    let err = AXUIElementCopyActionNames(element, &actions)
    guard err == .success, let names = actions as? [String] else { return [] }
    return names
}

func cleanValue(_ value: Any?) -> Any? {
    guard let value = value else { return nil }
    let cfValue = value as CFTypeRef
    if CFGetTypeID(cfValue) == AXValueGetTypeID() {
        let axValue = value as! AXValue
        switch AXValueGetType(axValue) {
        case .cgPoint:
            var point = CGPoint.zero
            if AXValueGetValue(axValue, .cgPoint, &point) {
                return ["x": Double(point.x), "y": Double(point.y)]
            }
        case .cgSize:
            var size = CGSize.zero
            if AXValueGetValue(axValue, .cgSize, &size) {
                return ["width": Double(size.width), "height": Double(size.height)]
            }
        case .cgRect:
            var rect = CGRect.zero
            if AXValueGetValue(axValue, .cgRect, &rect) {
                return [
                    "x": Double(rect.origin.x), "y": Double(rect.origin.y),
                    "width": Double(rect.size.width), "height": Double(rect.size.height),
                ]
            }
        case .cfRange:
            var range = CFRange(location: 0, length: 0)
            if AXValueGetValue(axValue, .cfRange, &range) {
                return ["location": range.location, "length": range.length]
            }
        default:
            break
        }
        return nil
    }
    if CFGetTypeID(cfValue) == AXUIElementGetTypeID() { return nil }
    if let string = value as? String {
        return string.count > 500 ? String(string.prefix(500)) : string
    }
    if let number = value as? NSNumber {
        return CFGetTypeID(number) == CFBooleanGetTypeID() ? Bool(truncating: number) : number
    }
    return String(describing: value)
}

func textFromAttributes(_ attrs: [String: Any]) -> String? {
    for key in ["AXTitle", "AXDescription", "AXValue", "AXPlaceholderValue", "AXHelp", "AXIdentifier"] {
        if let text = attrs[key] as? String,
           !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return text
        }
    }
    return nil
}

func children(_ element: AXUIElement) -> [AXUIElement] {
    var result: [AXUIElement] = []
    var seen = Set<String>()
    for attr in childAttributes {
        let (ok, value, _) = copyAttribute(element, attr)
        guard ok, let values = value as? [Any] else { continue }
        for item in values {
            let cfItem = item as CFTypeRef
            if CFGetTypeID(cfItem) == AXUIElementGetTypeID() {
                let key = String(describing: item)
                if seen.insert(key).inserted {
                    result.append(item as! AXUIElement)
                }
            }
        }
    }
    return result
}

func dumpNode(_ element: AXUIElement, depth: Int, cfg: Config, remaining: inout Int, visited: inout Set<UInt>) -> [String: Any]? {
    if remaining <= 0 { return nil }
    let identity = CFHash(element)
    if visited.contains(identity) { return nil }
    visited.insert(identity)
    remaining -= 1

    var attrs: [String: Any] = [:]
    var attrErrors: [String: String] = [:]
    for attr in coreAttributes {
        let (ok, value, err) = copyAttribute(element, attr)
        if ok, let cleaned = cleanValue(value) {
            attrs[attr] = cleaned
        } else if attr == "AXPosition" || attr == "AXSize" {
            attrErrors[attr] = String(describing: err)
        }
    }

    var node: [String: Any] = [
        "role": attrs["AXRole"] ?? NSNull(),
        "subrole": attrs["AXSubrole"] ?? NSNull(),
        "text": textFromAttributes(attrs) ?? NSNull(),
        "role_description": attrs["AXRoleDescription"] ?? NSNull(),
        "identifier": attrs["AXIdentifier"] ?? NSNull(),
        "enabled": attrs["AXEnabled"] ?? NSNull(),
        "focused": attrs["AXFocused"] ?? NSNull(),
        "position": attrs["AXPosition"] ?? NSNull(),
        "size": attrs["AXSize"] ?? NSNull(),
        "actions": actionNames(element),
    ]
    if !attrErrors.isEmpty {
        node["attribute_errors"] = attrErrors
    }
    if depth < cfg.maxDepth && remaining > 0 {
        var dumpedChildren: [[String: Any]] = []
        for child in children(element) {
            if let dumped = dumpNode(child, depth: depth + 1, cfg: cfg, remaining: &remaining, visited: &visited) {
                dumpedChildren.append(dumped)
            }
            if remaining <= 0 { break }
        }
        if !dumpedChildren.isEmpty {
            node["children"] = dumpedChildren
        }
    }
    return node
}

func selectWindowRoots(_ appElement: AXUIElement, cfg: Config) -> ([AXUIElement], [String: Any]) {
    let (ok, value, err) = copyAttribute(appElement, "AXWindows")
    guard ok, let values = value as? [Any] else {
        return ([appElement], ["selected": "application", "windows_error": String(describing: err)])
    }
    let windows = values.compactMap { item -> AXUIElement? in
        let cfItem = item as CFTypeRef
        return CFGetTypeID(cfItem) == AXUIElementGetTypeID() ? (item as! AXUIElement) : nil
    }
    var meta: [String: Any] = ["window_count": windows.count]
    if let target = cfg.windowBounds {
        let matches = windows.filter { window in
            guard let pos = cleanValue(copyAttribute(window, "AXPosition").1) as? [String: Double],
                  let size = cleanValue(copyAttribute(window, "AXSize").1) as? [String: Double] else { return false }
            let dx = abs((pos["x"] ?? -999999) - target[0])
            let dy = abs((pos["y"] ?? -999999) - target[1])
            let dw = abs((size["width"] ?? -999999) - target[2])
            let dh = abs((size["height"] ?? -999999) - target[3])
            return dx <= 8 && dy <= 8 && dw <= 16 && dh <= 16
        }
        meta["selected"] = "window_bounds"
        meta["window_bounds"] = target
        if !matches.isEmpty { return ([matches[0]], meta) }
    }
    if let title = cfg.windowTitle, !title.isEmpty {
        let needle = title.lowercased()
        let filtered = windows.filter { window in
            let (ok, value, _) = copyAttribute(window, "AXTitle")
            return ok && String(describing: value ?? "").lowercased().contains(needle)
        }
        meta["selected"] = "window_title"
        meta["window_title"] = title
        return (filtered, meta)
    }
    meta["selected"] = "window_index"
    meta["window_index"] = cfg.windowIndex
    if cfg.windowIndex >= 0 && cfg.windowIndex < windows.count {
        return ([windows[cfg.windowIndex]], meta)
    }
    return ([], meta)
}

let cfg = parseArgs()
let trusted = AXIsProcessTrusted()
let appElement = AXUIElementCreateApplication(cfg.pid)
let (rootsToDump, selection) = selectWindowRoots(appElement, cfg: cfg)
if rootsToDump.isEmpty {
    emitJSON([
        "ok": false,
        "error": "window_not_found",
        "trusted": trusted,
        "pid": Int(cfg.pid),
        "selection": selection,
    ], exitCode: 1)
}

var remaining = cfg.maxNodes
var roots: [[String: Any]] = []
var visited = Set<UInt>()
for root in rootsToDump {
    if let dumped = dumpNode(root, depth: 0, cfg: cfg, remaining: &remaining, visited: &visited) {
        roots.append(dumped)
    }
    if remaining <= 0 { break }
}

emitJSON([
    "ok": true,
    "trusted": trusted,
    "pid": Int(cfg.pid),
    "selection": selection,
    "limits": [
        "max_depth": cfg.maxDepth,
        "max_nodes": cfg.maxNodes,
        "nodes_emitted": cfg.maxNodes - remaining,
    ],
    "roots": roots,
])
