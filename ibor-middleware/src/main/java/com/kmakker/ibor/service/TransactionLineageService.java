package com.kmakker.ibor.service;

import com.kmakker.ibor.dto.LotDTO;
import com.kmakker.ibor.dto.PositionDetailDTO;
import com.kmakker.ibor.dto.TransactionDTO;
import com.kmakker.ibor.jodi.JodiTransactionLineageRepository;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

@Service
public class TransactionLineageService {
    private final JodiTransactionLineageRepository transactionLineageRepository;

    public TransactionLineageService(JodiTransactionLineageRepository transactionLineageRepository) {
        this.transactionLineageRepository = transactionLineageRepository;
    }

    public PositionDetailDTO getDetail(LocalDate asOf, String portfolioCode, String instrumentCode, String lotView) {
        if (asOf == null) throw new IllegalArgumentException("asOf must be provided.");
        if (portfolioCode == null || portfolioCode.isBlank()) throw new IllegalArgumentException("portfolioCode must be provided.");
        if (instrumentCode == null || instrumentCode.isBlank()) throw new IllegalArgumentException("instrumentCode must be provided.");

        var header = transactionLineageRepository.fetchHeader(asOf, portfolioCode, instrumentCode);
        if (header == null) {
            return new PositionDetailDTO(
                    asOf, portfolioCode, instrumentCode, null, null,
                    BigDecimal.ZERO, BigDecimal.ZERO, BigDecimal.ZERO,
                    "USD", BigDecimal.ZERO,
                    "NONE", List.of(), List.of());
        }
        List<TransactionDTO> transactions = transactionLineageRepository.fetchTransactions(asOf, portfolioCode, instrumentCode);

        List<LotDTO> lots = switch (lotView == null ? "NONE" : lotView.toUpperCase()) {
            case "FIFO" -> fifoLots(transactions);
            case "LIFO" -> lifoLots(transactions);
            case "AVG", "AVERAGE" -> avgLots(transactions);
            default -> List.of();
        };

        String lottingMethod = (lotView == null)? "NONE" : lotView.toUpperCase();

        return new PositionDetailDTO(
                header.asOf(),
                header.portfolioCode(),
                header.instrumentCode(),
                header.instrumentName(),
                header.instrumentType(),
                header.netQty(),
                header.price(),
                header.marketValue(),
                header.currency(),
                header.unrealizedPnl(),
                lottingMethod,
                transactions,
                lots
        );
    }

    private List<LotDTO> fifoLots(List<TransactionDTO> transactions) {
        // TODO: implement FIFO lot logic
        return new ArrayList<>();
    }

    private List<LotDTO> lifoLots(List<TransactionDTO> transactions) {
        // TODO: implement LIFO lot logic
        return new ArrayList<>();
    }

    private List<LotDTO> avgLots(List<TransactionDTO> transactions) {
        // TODO: implement AVG lot logic
        return new ArrayList<>();
    }
}
