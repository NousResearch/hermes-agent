import { api } from "@/lib/api";

export const restartGatewayFromChannelsPage = () => api.restartGateway();
export const restartGatewayAfterTelegramOnboarding = () => api.restartGateway();
export const restartGatewayFromSystemPage = () => api.restartGateway();
export const restartGatewayFromWebhooksPage = () => api.restartGateway();
export const updateHermesFromSystemPage = () => api.updateHermes();
