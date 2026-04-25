package com.kmakker.ibor.dto;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;

public record PositionDetailDTO (
    LocalDate asOf,
    String portfolioCode,
    String instrumentCode,
    String instrumentName,
    String instrumentType,
    BigDecimal netQty,
    BigDecimal price,
    BigDecimal marketValue,
    String currency,
    BigDecimal unrealizedPnl,
    String lottingMethod,        // FIFO / LIFO / AVG / NONE
    List<TransactionDTO> transactions,
    List<LotDTO> lots
) {}
