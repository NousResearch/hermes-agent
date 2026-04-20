import { jsx as _jsx } from "react/jsx-runtime";
import { createContext, useContext } from 'react';
const GatewayContext = createContext(null);
export function GatewayProvider({ children, value }) {
    return _jsx(GatewayContext.Provider, { value: value, children: children });
}
export function useGateway() {
    const value = useContext(GatewayContext);
    if (!value) {
        throw new Error('GatewayContext missing');
    }
    return value;
}
