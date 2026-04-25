package com.kmakker.ibor.service;

import com.kmakker.ibor.dto.PositionDetailDTO;
import com.kmakker.ibor.dto.TransactionDTO;
import com.kmakker.ibor.jodi.JodiTransactionLineageRepository;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;

public class TransactionLineageServiceTest {
    @Test
    void getDetail_returns_transactions() {
        var repo = Mockito.mock(JodiTransactionLineageRepository.class);
        var service = new TransactionLineageService(repo);

        var header = new PositionDetailDTO(
                LocalDate.now(), "ALPHA", "IBM", "IBM Corp.", "EQUITY",
                BigDecimal.valueOf(100), BigDecimal.valueOf(120),
                BigDecimal.valueOf(12000), "USD", null, "NONE",
                List.of(), List.of());

        var transactions = new TransactionDTO("TRADE", "T1", null, "BUY",
                BigDecimal.valueOf(100), BigDecimal.valueOf(120),
                BigDecimal.valueOf(12000), "GS", "CORE", null);

        when(repo.fetchHeader(Mockito.any(), Mockito.anyString(), Mockito.anyString())).thenReturn(header);
        when (repo.fetchTransactions(Mockito.any(), Mockito.anyString(), Mockito.anyString())).thenReturn(List.of(transactions));

        var result = service.getDetail(LocalDate.now(), "ALPHA", "IBM", "NONE");
        assertThat(result.transactions().size()).isEqualTo(1);
        assertThat(header.portfolioCode()).isEqualTo("ALPHA");
    }
}
