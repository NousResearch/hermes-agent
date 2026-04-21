import ApplicationServices
import CoreGraphics
import Foundation

enum PointerError: Error, CustomStringConvertible {
    case usage(String)
    case invalid(String)
    case event(String)

    var description: String {
        switch self {
        case .usage(let message), .invalid(let message), .event(let message):
            return message
        }
    }
}

func argumentValue(_ args: [String], _ flag: String) -> String? {
    guard let index = args.firstIndex(of: flag), index + 1 < args.count else { return nil }
    return args[index + 1]
}

func requiredInt(_ args: [String], _ flag: String) throws -> Int {
    guard let raw = argumentValue(args, flag) else {
        throw PointerError.usage("Missing required flag: \(flag)")
    }
    guard let value = Int(raw) else {
        throw PointerError.invalid("Flag \(flag) must be an integer")
    }
    return value
}

func optionalInt(_ args: [String], _ flag: String) throws -> Int? {
    guard let raw = argumentValue(args, flag) else { return nil }
    guard let value = Int(raw) else {
        throw PointerError.invalid("Flag \(flag) must be an integer")
    }
    return value
}

func requiredString(_ args: [String], _ flag: String) throws -> String {
    guard let raw = argumentValue(args, flag) else {
        throw PointerError.usage("Missing required flag: \(flag)")
    }
    let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty else {
        throw PointerError.invalid("Flag \(flag) cannot be empty")
    }
    return text
}

func post(_ event: CGEvent?, description: String) throws {
    guard let event else {
        throw PointerError.event("Failed to create \(description) event")
    }
    event.post(tap: .cghidEventTap)
}

func moveCursor(to point: CGPoint) throws {
    try post(
        CGEvent(mouseEventSource: nil, mouseType: .mouseMoved, mouseCursorPosition: point, mouseButton: .left),
        description: "mouse move"
    )
}

enum PointerButton: String {
    case left
    case right
    case middle

    var cgButton: CGMouseButton {
        switch self {
        case .left: return .left
        case .right: return .right
        case .middle: return .center
        }
    }

    var downType: CGEventType {
        switch self {
        case .left: return .leftMouseDown
        case .right: return .rightMouseDown
        case .middle: return .otherMouseDown
        }
    }

    var upType: CGEventType {
        switch self {
        case .left: return .leftMouseUp
        case .right: return .rightMouseUp
        case .middle: return .otherMouseUp
        }
    }

    var draggedType: CGEventType {
        switch self {
        case .left: return .leftMouseDragged
        case .right: return .rightMouseDragged
        case .middle: return .otherMouseDragged
        }
    }
}

func pointerButton(_ raw: String) throws -> PointerButton {
    guard let button = PointerButton(rawValue: raw.lowercased()) else {
        throw PointerError.invalid("Unsupported button: \(raw)")
    }
    return button
}

func performClick(args: [String]) throws -> [String: Any] {
    let x = try requiredInt(args, "--x")
    let y = try requiredInt(args, "--y")
    let button = try pointerButton(requiredString(args, "--button"))
    let count = max(1, try requiredInt(args, "--count"))
    let point = CGPoint(x: x, y: y)
    try moveCursor(to: point)
    usleep(15000)
    for clickIndex in 1...count {
        let state = Int64(clickIndex)
        let down = CGEvent(mouseEventSource: nil, mouseType: button.downType, mouseCursorPosition: point, mouseButton: button.cgButton)
        down?.setIntegerValueField(.mouseEventClickState, value: state)
        try post(down, description: "mouse down")
        let up = CGEvent(mouseEventSource: nil, mouseType: button.upType, mouseCursorPosition: point, mouseButton: button.cgButton)
        up?.setIntegerValueField(.mouseEventClickState, value: state)
        try post(up, description: "mouse up")
        usleep(15000)
    }
    return [
        "x": x,
        "y": y,
        "button": button.rawValue,
        "click_count": count,
    ]
}

func performScroll(args: [String]) throws -> [String: Any] {
    let deltaY = try requiredInt(args, "--delta-y")
    let x = try optionalInt(args, "--x")
    let y = try optionalInt(args, "--y")
    if let x, let y {
        try moveCursor(to: CGPoint(x: x, y: y))
        usleep(15000)
    }
    let event = CGEvent(scrollWheelEvent2Source: nil, units: .line, wheelCount: 1, wheel1: Int32(deltaY), wheel2: 0, wheel3: 0)
    try post(event, description: "scroll")
    var payload: [String: Any] = ["delta_y": deltaY]
    if let x { payload["x"] = x }
    if let y { payload["y"] = y }
    return payload
}

func performDrag(args: [String]) throws -> [String: Any] {
    let startX = try requiredInt(args, "--start-x")
    let startY = try requiredInt(args, "--start-y")
    let endX = try requiredInt(args, "--end-x")
    let endY = try requiredInt(args, "--end-y")
    let startPoint = CGPoint(x: startX, y: startY)
    let endPoint = CGPoint(x: endX, y: endY)
    try moveCursor(to: startPoint)
    usleep(15000)
    try post(CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: startPoint, mouseButton: .left), description: "drag down")
    usleep(15000)
    try post(CGEvent(mouseEventSource: nil, mouseType: .leftMouseDragged, mouseCursorPosition: endPoint, mouseButton: .left), description: "drag move")
    usleep(15000)
    try post(CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: endPoint, mouseButton: .left), description: "drag up")
    return [
        "start_x": startX,
        "start_y": startY,
        "end_x": endX,
        "end_y": endY,
    ]
}

func emit(_ payload: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]) {
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data("\n".utf8))
    }
}

do {
    let args = Array(CommandLine.arguments.dropFirst())
    guard let action = args.first else {
        throw PointerError.usage("Usage: mac-pointer-action <click|scroll|drag> [flags]")
    }
    let remainder = Array(args.dropFirst())
    let result: [String: Any]
    switch action {
    case "click":
        result = try performClick(args: remainder)
    case "scroll":
        result = try performScroll(args: remainder)
    case "drag":
        result = try performDrag(args: remainder)
    default:
        throw PointerError.usage("Unknown action: \(action)")
    }
    var payload = result
    payload["success"] = true
    emit(payload)
} catch {
    emit(["success": false, "error": String(describing: error)])
    exit(1)
}
