package com.kmakker.ibor.infra;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

@Component
@Order(1)
public class ApiKeyFilter extends OncePerRequestFilter {

    @Value("${API_KEY:}")
    private String expectedApiKey;

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain chain) throws ServletException, IOException {
        String path = request.getRequestURI();

        // Skip auth for health check
        if (path.startsWith("/actuator")) {
            chain.doFilter(request, response);
            return;
        }

        // If no key configured, allow all (local dev)
        if (expectedApiKey == null || expectedApiKey.isBlank()) {
            chain.doFilter(request, response);
            return;
        }

        String providedKey = request.getHeader("X-API-Key");
        if (!expectedApiKey.equals(providedKey)) {
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            response.setContentType("application/json");
            response.getWriter().write("{\"error\":\"Invalid or missing API key\"}");
            return;
        }

        chain.doFilter(request, response);
    }
}
